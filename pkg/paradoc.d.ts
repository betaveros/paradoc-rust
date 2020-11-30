/* tslint:disable */
/* eslint-disable */
/**
* @param {string} code
* @param {string} input
* @returns {WasmOutputs}
*/
export function encapsulated_eval(code: string, input: string): WasmOutputs;
/**
*/
export class WasmOutputs {
  free(): void;
/**
* @returns {string}
*/
  get_output(): string;
/**
* @returns {string}
*/
  get_error(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmoutputs_free: (a: number) => void;
  readonly wasmoutputs_get_output: (a: number, b: number) => void;
  readonly wasmoutputs_get_error: (a: number, b: number) => void;
  readonly encapsulated_eval: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
        