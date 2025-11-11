// Copyright 2018-2025 the Deno authors. MIT license.

/**
 * Utilities for working with Deno's readers, writers, and web streams.
 *
 * `Reader` and `Writer` interfaces are deprecated in Deno, and so many of these
 * utilities are also deprecated. Consider using web streams instead.
 *
 * ```ts ignore
 * import { toReadableStream, toWritableStream } from "@std/io";
 *
 * await toReadableStream(Deno.stdin)
 *   .pipeTo(toWritableStream(Deno.stdout));
 * ```
 *
 * @module
 */

export * from "./buffer.js";
export * from "./copy.js";
export * from "./iterate_reader.js";
export * from "./read_all.js";
export * from "./reader_from_stream_reader.js";
export * from "./to_readable_stream.js";
export * from "./to_writable_stream.js";
export * from "./types.js";
export * from "./write_all.js";
