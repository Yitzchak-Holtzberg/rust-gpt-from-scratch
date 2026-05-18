//! Localhost training + inference server. Open http://127.0.0.1:8080/ after
//! launching to use the live training dashboard. All logic lives in the
//! `nano_gpt_rs::server` module — this binary is just the entry point.
//!
//! Run with: `cargo run --release --features cuda --bin serve`

fn main() {
    nano_gpt_rs::server::run();
}
