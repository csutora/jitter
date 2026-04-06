{ pkgs ? import <nixpkgs> {} }:
let
  aarch64Musl = pkgs.pkgsCross.aarch64-multiplatform-musl;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_22
    # Cross-compiler for static aarch64-linux-musl conformance test binaries
    aarch64Musl.stdenv.cc
    # Build tools for LTP cross-compilation
    autoconf automake libtool pkg-config gnumake
  ];
}
