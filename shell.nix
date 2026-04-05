{ pkgs ? import <nixpkgs> {} }:
let
  aarch64Cross = pkgs.pkgsCross.aarch64-multiplatform;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_22
    # Cross-compiler for aarch64-linux conformance test binaries
    aarch64Cross.stdenv.cc
  ];
}
