//! The moa's face, the wordmark, colour, and the boxes `info` / `doctor` draw.
//!
//! The moa was New Zealand's giant flightless bird — a ratite that moved in
//! flocks. moa finds the flocks hiding in a graph.

use std::sync::atomic::{AtomicBool, Ordering};

pub const CATCHPHRASE: &str = "Extinct bird. Living clusters.";
pub const TAGLINE: &str = "reference-faithful Markov clustering · MinHash graphs · 100% Rust";

static COLOR: AtomicBool = AtomicBool::new(false);

/// Decide colour once: `--color always|never|auto`, `NO_COLOR`, TTY detection.
pub fn init_color(mode: &str, is_tty: bool) {
    let on = match mode {
        "always" => true,
        "never" => false,
        _ => {
            is_tty
                && std::env::var_os("NO_COLOR").is_none()
                && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
        }
    };
    COLOR.store(on, Ordering::Relaxed);
}

pub fn color() -> bool {
    COLOR.load(Ordering::Relaxed)
}

/// 256-colour foreground (falls back to plain text when colour is off).
pub fn fg(code: u8, s: &str) -> String {
    if color() {
        format!("\x1b[38;5;{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
pub fn bold(s: &str) -> String {
    if color() {
        format!("\x1b[1m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
pub fn dim(s: &str) -> String {
    if color() {
        format!("\x1b[2m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

// Palette: moa feathers, amber beak, bush-green wordmark.
pub const BROWN: u8 = 137;
pub const AMBER: u8 = 214;
pub const CREAM: u8 = 230;
pub const GREEN: u8 = 78;
pub const TEAL: u8 = 37;
pub const RED: u8 = 203;
pub const GOLD: u8 = 221;
pub const GREY: u8 = 245;

/// The moa's face (crest, eyes, two-part beak). 15 columns × 6 rows.
pub const FACE: [&str; 6] = [
    "    ,_  _,     ",
    "  .' \\''/ '.   ",
    " /  (o)(o)  \\  ",
    " |    /\\    |  ",
    " \\    \\/    /  ",
    "  '-.____.-'   ",
];

/// "MOA" in FIGlet ANSI Shadow.
pub const WORD: [&str; 6] = [
    "███╗   ███╗ ██████╗  █████╗ ",
    "████╗ ████║██╔═══██╗██╔══██╗",
    "██╔████╔██║██║   ██║███████║",
    "██║╚██╔╝██║██║   ██║██╔══██║",
    "██║ ╚═╝ ██║╚██████╔╝██║  ██║",
    "╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝",
];

/// Teal → green gradient, one shade per wordmark row.
const WORD_SHADES: [u8; 6] = [30, 36, 37, 43, 78, 114];

fn paint_face(row: usize, line: &str) -> String {
    if !color() {
        return line.to_string();
    }
    let mut out = String::new();
    for ch in line.chars() {
        let s = ch.to_string();
        out += &match ch {
            'o' => bold(&fg(CREAM, &s)),
            '/' | '\\' if row == 3 || row == 4 => fg(AMBER, &s), // the beak
            '(' | ')' => fg(CREAM, &s),
            ' ' => s,
            _ => fg(BROWN, &s),
        };
    }
    out
}

/// The logo: face beside the wordmark.
pub fn logo() -> String {
    let mut s = String::new();
    for r in 0..6 {
        s.push_str("  ");
        s.push_str(&paint_face(r, FACE[r]));
        s.push_str("  ");
        s.push_str(&bold(&fg(WORD_SHADES[r], WORD[r])));
        s.push('\n');
    }
    s
}

/// Logo + version/tagline + catchphrase + an optional subtitle.
pub fn banner(subtitle: &str) -> String {
    let ver = format!("v{}", crate::VERSION);
    let mut s = logo();
    s.push_str(&format!("  {}  {}\n", bold(&fg(GOLD, &ver)), dim(TAGLINE)));
    s.push_str(&format!("  {}\n", fg(BROWN, &format!("“{CATCHPHRASE}”"))));
    if !subtitle.is_empty() {
        s.push_str(&format!("  {}\n", fg(TEAL, subtitle)));
    }
    s
}

/// A tiny inline mascot for one-liners.
pub fn mini() -> String {
    format!("{}{}{}", fg(BROWN, "("), bold(&fg(CREAM, "o")), fg(BROWN, ")>"))
}

/// Display width, ignoring ANSI escapes (all glyphs we print are 1 column).
pub fn width(s: &str) -> usize {
    let mut w = 0;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            if c == 'm' {
                esc = false;
            }
        } else if c == '\x1b' {
            esc = true;
        } else {
            w += 1;
        }
    }
    w
}

pub fn pad(s: &str, w: usize) -> String {
    let n = width(s);
    if n >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - n))
    }
}

/// A rounded box with a title and free-form rows.
pub fn panel(title: &str, rows: &[String]) -> String {
    let inner = rows.iter().map(|r| width(r)).chain(std::iter::once(width(title) + 2)).max().unwrap_or(0).max(66);
    let mut s = String::new();
    let t = format!(" {} ", bold(&fg(GOLD, title)));
    s.push_str(&fg(GREY, "╭─"));
    s.push_str(&t);
    s.push_str(&fg(GREY, &format!("{}╮", "─".repeat(inner + 1 - width(&t)))));
    s.push('\n');
    for r in rows {
        s.push_str(&fg(GREY, "│ "));
        s.push_str(&pad(r, inner));
        s.push_str(&fg(GREY, " │"));
        s.push('\n');
    }
    s.push_str(&fg(GREY, &format!("╰{}╯", "─".repeat(inner + 2))));
    s.push('\n');
    s
}

/// `key   value` row for panels.
pub fn kv(k: &str, v: &str) -> String {
    format!("{}  {}", pad(&fg(TEAL, k), 18), v)
}

pub fn ok() -> String {
    bold(&fg(GREEN, "✔"))
}
pub fn bad() -> String {
    bold(&fg(RED, "✘"))
}
pub fn skip() -> String {
    fg(GOLD, "○")
}

/// Human-friendly durations.
pub fn dur(secs: f64) -> String {
    if secs < 1e-3 {
        format!("{:.0} µs", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.1} ms", secs * 1e3)
    } else if secs < 120.0 {
        format!("{secs:.2} s")
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

/// Thousands separators.
pub fn num(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
