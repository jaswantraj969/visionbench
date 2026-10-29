import { describe, expect, it } from "vitest";

import { bytes, ci, meanSd, num, pct, seconds, shortSha, signed } from "./format";

describe("format", () => {
  it("never renders a missing value as 0", () => {
    expect(num(null)).toBe("undefined");
    expect(num(undefined)).toBe("undefined");
    expect(num(Number.NaN)).toBe("undefined");
    expect(pct(null)).toBe("undefined");
    expect(meanSd(null, 0.1)).toBe("undefined");
    expect(ci(null, 0, 1)).toBe("undefined");
    expect(signed(null)).toBe("undefined");
    expect(seconds(null)).toBe("undefined");
    expect(bytes(null)).toBe("undefined");
    expect(num(0)).toBe("0.0000");
  });

  it("formats values", () => {
    expect(meanSd(0.5, 0.01)).toBe("0.5000 ± 0.0100");
    expect(meanSd(0.5, null)).toBe("0.5000");
    expect(ci(0.5, 0.4, 0.6)).toBe("0.5000 [0.4000, 0.6000]");
    expect(ci(0.5, null, 0.6)).toBe("0.5000 [CI undefined]");
    expect(signed(0.25, 2)).toBe("+0.25");
    expect(signed(-0.25, 2)).toBe("-0.25");
    expect(pct(0.234375)).toBe("23.4 %");
    expect(seconds(30)).toBe("30.0 s");
    expect(seconds(90)).toBe("1.5 min");
    expect(bytes(2048)).toBe("2.0 KiB");
    expect(bytes(3 * 1024 ** 2)).toBe("3.0 MiB");
    expect(shortSha("abcdef0123456789")).toBe("abcdef012345…");
    expect(shortSha(null)).toBe("none");
  });
});
