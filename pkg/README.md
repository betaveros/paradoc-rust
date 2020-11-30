# paradoc-rust

This is a reimplementation of my [tremendously overengineered golflang, Paradoc](https://github.com/betaveros/paradoc), in Rust.

I don't think this implementation will replace the original or even necessarily reach full feature parity and/or compatibility, just because it's so much more effort per feature. But we'll try to get close. The short reasons this exists are in some order:

1. I wanted something I could feasibly compile to WASM so people can run code without having to clone the repo to their computer and know their way around Python and the command line and such (not that this is implemented yet);
2. I wanted a faster implementation;
3. I wanted to learn Rust, so why not. (This is my first Rust project.)

I might expound on this somewhere later.

## extremely unscientific speed benchmark

My Advent of Code Day 5 Part 2 program takes this long to run on these interpreters:

- paradoc (python): 143s
- paradoc-rust (debug build): 65s
- paradoc-rust (release build): 3.1s
