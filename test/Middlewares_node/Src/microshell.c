/**
 ******************************************************************************
 * @file    microshell.c
 * @author  Daisy Berrios
 * @date    Jan 13, 2026
 * @brief   Minimal USB-CDC microshell (commands over USB_OTG_FS)
 ******************************************************************************
 */

#include "microshell.h"
#include <string.h>
#include <stdio.h>

/* If you use CubeMX USB CDC, include this to use CDC_Transmit_FS() */
#include "usbd_cdc_if.h"

/*- PRIVATE_Definitions ------------------------------------------------------*/
#define MS_PROMPT          "ws> "
#define MS_EOL             "\r\n"

#define MS_MIN(a,b)        ((a)<(b)?(a):(b))

/*- PRIVATE_Data -------------------------------------------------------------*/
static microshell_hooks_t s_hooks;

/* RX line accumulation */
static char     s_line[MICROSHELL_RX_BUF_SZ];
static uint16_t s_line_len = 0;
static bool     s_need_prompt = true;

/* TX ring buffer (for CDC busy-safe printing) */
static uint8_t  s_tx[MICROSHELL_TX_BUF_SZ];
static uint16_t s_tx_w = 0;
static uint16_t s_tx_r = 0;
static bool     s_tx_inflight = false;

/*- PRIVATE_Functions --------------------------------------------------------*/
static uint16_t tx_used(void)
{
  if (s_tx_w >= s_tx_r) return (uint16_t)(s_tx_w - s_tx_r);
  return (uint16_t)(MICROSHELL_TX_BUF_SZ - (s_tx_r - s_tx_w));
}

static uint16_t tx_free(void)
{
  /* keep one byte empty to distinguish full/empty */
  return (uint16_t)((MICROSHELL_TX_BUF_SZ - 1) - tx_used());
}

static void tx_push_bytes(const uint8_t *p, uint16_t n)
{
  while (n)
  {
    uint16_t space = tx_free();
    if (space == 0) break;

    uint16_t chunk = MS_MIN(space, n);

    /* write chunk possibly wrapping */
    uint16_t to_end = (uint16_t)(MICROSHELL_TX_BUF_SZ - s_tx_w);
    uint16_t w1 = MS_MIN(to_end, chunk);
    memcpy(&s_tx[s_tx_w], p, w1);
    s_tx_w = (uint16_t)((s_tx_w + w1) % MICROSHELL_TX_BUF_SZ);

    uint16_t w2 = (uint16_t)(chunk - w1);
    if (w2)
    {
      memcpy(&s_tx[s_tx_w], p + w1, w2);
      s_tx_w = (uint16_t)((s_tx_w + w2) % MICROSHELL_TX_BUF_SZ);
    }

    p += chunk;
    n -= chunk;
  }
}

static uint16_t tx_peek_contiguous(uint8_t *out, uint16_t maxlen)
{
  uint16_t used = tx_used();
  if (used == 0) return 0;

  uint16_t to_end = (uint16_t)(MICROSHELL_TX_BUF_SZ - s_tx_r);
  uint16_t n = MS_MIN(used, to_end);
  n = MS_MIN(n, maxlen);

  memcpy(out, &s_tx[s_tx_r], n);
  return n;
}

static void tx_pop(uint16_t n)
{
  s_tx_r = (uint16_t)((s_tx_r + n) % MICROSHELL_TX_BUF_SZ);
}

static void ms_puts(const char *s)
{
  if (!s) return;
  tx_push_bytes((const uint8_t*)s, (uint16_t)strlen(s));
}

static void ms_puteol(void)
{
  ms_puts(MS_EOL);
}

static void ms_ok(void)
{
  ms_puts("OK");
  ms_puteol();
}

static void ms_err(const char *msg)
{
  ms_puts("ERR: ");
  ms_puts(msg ? msg : "unknown");
  ms_puteol();
}

/* return true if integer parsed fully */
static bool parse_u32(const char *s, uint32_t *out)
{
  if (!s || !*s || !out) return false;
  uint32_t v = 0;

  while (*s == ' ' || *s == '\t') s++;
  if (*s < '0' || *s > '9') return false;

  while (*s >= '0' && *s <= '9')
  {
    uint32_t nv = (uint32_t)(v * 10u + (uint32_t)(*s - '0'));
    if (nv < v) return false; /* overflow */
    v = nv;
    s++;
  }

  while (*s == ' ' || *s == '\t') s++;
  if (*s != '\0') return false;

  *out = v;
  return true;
}

static bool parse_datetime(const char *date_s, const char *time_s, microshell_datetime_t *dt)
{
  if (!date_s || !time_s || !dt) return false;

  /* Expect YYYY-MM-DD and HH:MM:SS */
  int y=0, mo=0, d=0, h=0, mi=0, se=0;
  if (sscanf(date_s, "%d-%d-%d", &y, &mo, &d) != 3) return false;
  if (sscanf(time_s, "%d:%d:%d", &h, &mi, &se) != 3) return false;

  if (y < 2000 || y > 2099) return false;
  if (mo < 1 || mo > 12) return false;
  if (d < 1 || d > 31) return false;
  if (h < 0 || h > 23) return false;
  if (mi < 0 || mi > 59) return false;
  if (se < 0 || se > 59) return false;

  dt->year  = (uint16_t)y;
  dt->month = (uint8_t)mo;
  dt->day   = (uint8_t)d;
  dt->hour  = (uint8_t)h;
  dt->min   = (uint8_t)mi;
  dt->sec   = (uint8_t)se;
  return true;
}

static int tokenize(char *line, char **argv, int max_tokens)
{
  int argc = 0;
  char *p = line;

  while (*p && argc < max_tokens)
  {
    while (*p == ' ' || *p == '\t') p++;
    if (!*p) break;

    argv[argc++] = p;

    while (*p && *p != ' ' && *p != '\t') p++;
    if (*p)
    {
      *p = '\0';
      p++;
    }
  }
  return argc;
}

static void cmd_help(void)
{
  ms_puts("Commands:");
  ms_puteol();

  ms_puts("  help");
  ms_puteol();

  ms_puts("  rtc get");
  ms_puteol();
  ms_puts("  rtc set YYYY-MM-DD HH:MM:SS");
  ms_puteol();

  ms_puts("  wake get");
  ms_puteol();
  ms_puts("  wake set <seconds>");
  ms_puteol();

  ms_puts("  txint get");
  ms_puteol();
  ms_puts("  txint set <seconds>");
  ms_puteol();

  ms_puts("  data list");
  ms_puteol();
  ms_puts("  data read <index>");
  ms_puteol();
  ms_puts("  data clear");
  ms_puteol();

  ms_puts("  status");
  ms_puteol();
  ms_puts("  txtest");
  ms_puteol();
  ms_puts("  regs");
  ms_puteol();
}

static void cmd_rtc(int argc, char **argv)
{
  if (argc < 2) { ms_err("rtc get|set"); return; }

  if (strcmp(argv[1], "get") == 0)
  {
    if (!s_hooks.rtc_get) { ms_err("rtc_get not hooked"); return; }
    microshell_datetime_t dt;
    if (!s_hooks.rtc_get(&dt)) { ms_err("rtc_get failed"); return; }

    char buf[48];
    snprintf(buf, sizeof(buf), "%04u-%02u-%02u %02u:%02u:%02u",
             dt.year, dt.month, dt.day, dt.hour, dt.min, dt.sec);
    ms_puts(buf); ms_puteol();
    return;
  }

  if (strcmp(argv[1], "set") == 0)
  {
    if (!s_hooks.rtc_set) { ms_err("rtc_set not hooked"); return; }
    if (argc < 4) { ms_err("rtc set YYYY-MM-DD HH:MM:SS"); return; }

    microshell_datetime_t dt;
    if (!parse_datetime(argv[2], argv[3], &dt)) { ms_err("bad datetime"); return; }
    if (!s_hooks.rtc_set(&dt)) { ms_err("rtc_set failed"); return; }
    ms_ok();
    return;
  }

  ms_err("rtc get|set");
}

static void cmd_interval(const char *name, int argc, char **argv,
                         bool (*get_s)(uint32_t*), bool (*set_s)(uint32_t))
{
  if (argc < 2)
  {
    ms_puts("ERR: "); ms_puts(name); ms_puts(" get|set"); ms_puteol();
    return;
  }

  if (strcmp(argv[1], "get") == 0)
  {
    if (!get_s) { ms_err("get not hooked"); return; }
    uint32_t s = 0;
    if (!get_s(&s)) { ms_err("get failed"); return; }
    char buf[32];
    snprintf(buf, sizeof(buf), "%lu", (unsigned long)s);
    ms_puts(buf); ms_puteol();
    return;
  }

  if (strcmp(argv[1], "set") == 0)
  {
    if (!set_s) { ms_err("set not hooked"); return; }
    if (argc < 3) { ms_err("set <seconds>"); return; }

    uint32_t s = 0;
    if (!parse_u32(argv[2], &s)) { ms_err("bad number"); return; }

    if (!set_s(s)) { ms_err("set failed"); return; }
    ms_ok();
    return;
  }

  ms_err("get|set");
}

static void cmd_data(int argc, char **argv)
{
  if (argc < 2) { ms_err("data list|read|clear"); return; }

  if (strcmp(argv[1], "list") == 0)
  {
    if (!s_hooks.data_list) { ms_err("data_list not hooked"); return; }
    if (!s_hooks.data_list()) { ms_err("data_list failed"); return; }
    return;
  }

  if (strcmp(argv[1], "read") == 0)
  {
    if (!s_hooks.data_read) { ms_err("data_read not hooked"); return; }
    if (argc < 3) { ms_err("data read <index>"); return; }
    uint32_t idx = 0;
    if (!parse_u32(argv[2], &idx)) { ms_err("bad index"); return; }
    if (!s_hooks.data_read(idx)) { ms_err("data_read failed"); return; }
    return;
  }

  if (strcmp(argv[1], "clear") == 0)
  {
    if (!s_hooks.data_clear) { ms_err("data_clear not hooked"); return; }
    if (!s_hooks.data_clear()) { ms_err("data_clear failed"); return; }
    ms_ok();
    return;
  }

  ms_err("data list|read|clear");
}

static void exec_line(char *line)
{
  char *argv[MICROSHELL_MAX_TOKENS] = {0};
  int argc = tokenize(line, argv, MICROSHELL_MAX_TOKENS);

  if (argc == 0) return;

  if (strcmp(argv[0], "help") == 0 || strcmp(argv[0], "?") == 0)
  {
    cmd_help();
    return;
  }

  if (strcmp(argv[0], "rtc") == 0)
  {
    cmd_rtc(argc, argv);
    return;
  }

  if (strcmp(argv[0], "wake") == 0)
  {
    cmd_interval("wake", argc, argv, s_hooks.wakeup_get_s, s_hooks.wakeup_set_s);
    return;
  }

  if (strcmp(argv[0], "txint") == 0)
  {
    cmd_interval("txint", argc, argv, s_hooks.txint_get_s, s_hooks.txint_set_s);
    return;
  }

  if (strcmp(argv[0], "data") == 0)
  {
    cmd_data(argc, argv);
    return;
  }

  /* Custom hook */
  if (s_hooks.custom_line && s_hooks.custom_line(argc, argv))
  {
    return; /* user handled it */
  }

  ms_err("unknown cmd (try help)");
}

/*- PUBLIC_API ---------------------------------------------------------------*/
void microshell_init(const microshell_hooks_t *hooks)
{
  memset(&s_hooks, 0, sizeof(s_hooks));
  if (hooks) s_hooks = *hooks;

  s_line_len = 0;
  s_need_prompt = true;

  /* greeting */
  ms_puts(MS_EOL "Microshell (USB-CDC) ready. Type 'help'." MS_EOL);
  microshell_prompt();
}

void microshell_prompt(void)
{
  ms_puts(MS_PROMPT);
  s_need_prompt = false;
}

void microshell_print(const char *s)
{
  ms_puts(s);
}

void microshell_printf(const char *fmt, ...)
{
  char buf[160];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  ms_puts(buf);
}

/* Feed bytes; builds a line and executes on CR/LF */
void microshell_rx_bytes(const uint8_t *data, uint16_t len)
{
  if (!data || !len) return;

  for (uint16_t i = 0; i < len; i++)
  {
    char c = (char)data[i];

    /* handle CR/LF -> execute */
    if (c == '\r' || c == '\n')
    {
      if (s_line_len > 0)
      {
        s_line[s_line_len] = '\0';
        ms_puteol();
        exec_line(s_line);
        s_line_len = 0;
      }
      s_need_prompt = true;
      continue;
    }

    /* basic backspace support */
    if (c == '\b' || c == 127)
    {
      if (s_line_len > 0) s_line_len--;
      continue;
    }

    /* store */
    if (s_line_len < (MICROSHELL_RX_BUF_SZ - 1))
    {
      s_line[s_line_len++] = c;
    }
    else
    {
      /* line overflow -> reset */
      s_line_len = 0;
      ms_err("line too long");
      s_need_prompt = true;
    }
  }
}

/* Flush TX ring via CDC_Transmit_FS.
   Note: CDC_Transmit_FS returns USBD_OK or USBD_BUSY typically. */
void microshell_task(void)
{
  /* send prompt if needed */
  if (s_need_prompt)
  {
    microshell_prompt();
  }

  if (s_tx_inflight) return;

  uint16_t used = tx_used();
  if (used == 0) return;

  uint8_t tmp[64]; /* small packet */
  uint16_t n = tx_peek_contiguous(tmp, (uint16_t)sizeof(tmp));
  if (n == 0) return;

  int8_t st = CDC_Transmit_FS(tmp, n);
  if (st == 0) /* USBD_OK */
  {
    tx_pop(n);
    s_tx_inflight = false;
  }
  else
  {
    /* BUSY: just try again next call */
    s_tx_inflight = true;
    /* Clear inflight shortly (no callback here), so we don't stall forever.
       In practice, call microshell_task() often; BUSY will clear. */
    s_tx_inflight = false;
  }
}
