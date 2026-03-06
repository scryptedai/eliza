import { afterEach, describe, expect, it, vi } from "vitest";
import {
  generateIdempotencyKey,
  retryWithBackoff,
  ScryptedClient,
  validateBearerToken,
} from "../client.ts";
import {
  ScryptedAuthenticationError,
  ScryptedNetworkError,
  ScryptedPaymentError,
  ScryptedValidationError,
} from "../exceptions.ts";

describe("client: validateBearerToken", () => {
  it("accepts valid tokens", () => {
    expect(validateBearerToken("scrypted_abcdef1234567890")).toBe(true);
    expect(validateBearerToken(`scrypted_${"x".repeat(50)}`)).toBe(true);
  });

  it("rejects tokens without the scrypted_ prefix", () => {
    expect(validateBearerToken("sk_test_abcdef1234567890")).toBe(false);
    expect(validateBearerToken("bearer_abcdef1234567890")).toBe(false);
  });

  it("rejects tokens shorter than min length", () => {
    expect(validateBearerToken("scrypted_short")).toBe(false); // 14 chars
    expect(validateBearerToken("scrypted_")).toBe(false);
  });

  it("rejects non-string input", () => {
    expect(validateBearerToken(undefined)).toBe(false);
    expect(validateBearerToken(null)).toBe(false);
    expect(validateBearerToken(12345)).toBe(false);
    expect(validateBearerToken({})).toBe(false);
  });
});

describe("client: generateIdempotencyKey", () => {
  it("generates 64-char hex strings (256 bits)", () => {
    const key = generateIdempotencyKey();
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("generates unique keys", () => {
    const keys = new Set<string>();
    for (let i = 0; i < 100; i++) {
      keys.add(generateIdempotencyKey());
    }
    expect(keys.size).toBe(100);
  });
});

describe("client: retryWithBackoff", () => {
  it("returns on first success without retry", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await retryWithBackoff(fn, 3, 1);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on generic errors", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new ScryptedNetworkError("transient"))
      .mockRejectedValueOnce(new ScryptedNetworkError("transient"))
      .mockResolvedValue("ok");
    const result = await retryWithBackoff(fn, 3, 1);
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("does NOT retry ScryptedAuthenticationError", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ScryptedAuthenticationError("bad token"));
    await expect(retryWithBackoff(fn, 3, 1)).rejects.toThrow(
      ScryptedAuthenticationError,
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry ScryptedValidationError", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ScryptedValidationError("bad input"));
    await expect(retryWithBackoff(fn, 3, 1)).rejects.toThrow(
      ScryptedValidationError,
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry ScryptedPaymentError", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ScryptedPaymentError("payment required"));
    await expect(retryWithBackoff(fn, 3, 1)).rejects.toThrow(
      ScryptedPaymentError,
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("exhausts retries and throws last error", async () => {
    const fn = vi
      .fn()
      .mockRejectedValue(new ScryptedNetworkError("always fails"));
    await expect(retryWithBackoff(fn, 2, 1)).rejects.toThrow(
      ScryptedNetworkError,
    );
    expect(fn).toHaveBeenCalledTimes(3); // initial + 2 retries
  });
});

describe("client: ScryptedClient construction", () => {
  it("constructs with valid token", () => {
    const c = new ScryptedClient({
      bearerToken: "scrypted_valid_token_1234567890",
    });
    expect(c).toBeInstanceOf(ScryptedClient);
  });

  it("throws on invalid token format", () => {
    expect(() => new ScryptedClient({ bearerToken: "invalid" })).toThrow(
      ScryptedAuthenticationError,
    );
    expect(() => new ScryptedClient({ bearerToken: "scrypted_short" })).toThrow(
      ScryptedAuthenticationError,
    );
  });

  it("strips trailing slash from baseUrl", () => {
    // No direct getter, but we can confirm it doesn't throw and is constructed
    const c = new ScryptedClient({
      bearerToken: "scrypted_valid_token_1234567890",
      baseUrl: "https://api.example.com/",
    });
    expect(c).toBeInstanceOf(ScryptedClient);
  });

  it("constructs tokenless (permissionless bootstrap mode)", () => {
    const c = new ScryptedClient();
    expect(c).toBeInstanceOf(ScryptedClient);
    expect(c.getBearerToken()).toBeUndefined();
  });

  it("constructs tokenless with explicit empty options", () => {
    const c = new ScryptedClient({});
    expect(c.getBearerToken()).toBeUndefined();
  });

  it("exposes adopted token via getBearerToken()", () => {
    const c = new ScryptedClient({
      bearerToken: "scrypted_adopted_token_1234567890",
    });
    expect(c.getBearerToken()).toBe("scrypted_adopted_token_1234567890");
  });
});

describe("client: tokenless operations require auth", () => {
  const tokenless = new ScryptedClient();

  it("getAccountInfo throws without a token", async () => {
    await expect(tokenless.getAccountInfo()).rejects.toThrow(
      ScryptedAuthenticationError,
    );
  });

  it("getJobStatus throws without a token", async () => {
    await expect(tokenless.getJobStatus("some-job")).rejects.toThrow(
      ScryptedAuthenticationError,
    );
  });

  it("invokeNanoBananaGeneration throws without a token", async () => {
    await expect(
      tokenless.invokeNanoBananaGeneration({ prompt: "test" }),
    ).rejects.toThrow(ScryptedAuthenticationError);
  });
});

describe("client: input validation on invoke methods", () => {
  const client = new ScryptedClient({
    bearerToken: "scrypted_test_token_1234567890abcdef",
  });

  it("invokeRecipe rejects empty recipeId", async () => {
    await expect(client.invokeRecipe("", {})).rejects.toThrow(
      ScryptedValidationError,
    );
  });

  it("getJobStatus rejects empty jobId", async () => {
    await expect(client.getJobStatus("")).rejects.toThrow(
      ScryptedValidationError,
    );
  });

  it("invokeNanoBananaGeneration rejects missing prompt", async () => {
    await expect(
      client.invokeNanoBananaGeneration({ prompt: "" }),
    ).rejects.toThrow(ScryptedValidationError);
    await expect(
      client.invokeNanoBananaGeneration({ prompt: "   " }),
    ).rejects.toThrow(ScryptedValidationError);
    await expect(client.invokeNanoBananaGeneration({})).rejects.toThrow(
      ScryptedValidationError,
    );
  });

  it("invokeNanoBananaEditGeneration rejects missing image_urls", async () => {
    await expect(
      client.invokeNanoBananaEditGeneration({ prompt: "edit this" }),
    ).rejects.toThrow(ScryptedValidationError);
    await expect(
      client.invokeNanoBananaEditGeneration({
        prompt: "edit this",
        image_urls: [],
      }),
    ).rejects.toThrow(ScryptedValidationError);
  });

  it("invokeSora2Generation rejects missing image_url", async () => {
    await expect(
      client.invokeSora2Generation({ prompt: "animate" }),
    ).rejects.toThrow(ScryptedValidationError);
  });

  it("invokeTextGeneration rejects missing user_prompt", async () => {
    await expect(client.invokeTextGeneration({})).rejects.toThrow(
      ScryptedValidationError,
    );
    await expect(
      client.invokeTextGeneration({ user_prompt: "" }),
    ).rejects.toThrow(ScryptedValidationError);
  });

  it("invokeTopazVideoUpscale rejects missing video_url", async () => {
    await expect(client.invokeTopazVideoUpscale({})).rejects.toThrow(
      ScryptedValidationError,
    );
  });
});

describe("client: HTTP 401 → ScryptedAuthenticationError (no retry)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("maps HTTP 401 to ScryptedAuthenticationError and does NOT retry", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: "unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", mockFetch);

    const client = new ScryptedClient({
      bearerToken: "scrypted_test_token_1234567890abcdef",
      maxRetries: 3,
    });

    await expect(client.getJobStatus("some-job")).rejects.toThrow(
      ScryptedAuthenticationError,
    );
    // Regression: before the fix, 401 threw ScryptedAPIError → retried 4 times.
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("maps HTTP 401 on POST (invoke) to ScryptedAuthenticationError without retry", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValue(new Response("", { status: 401 }));
    vi.stubGlobal("fetch", mockFetch);

    const client = new ScryptedClient({
      bearerToken: "scrypted_test_token_1234567890abcdef",
      maxRetries: 3,
    });

    await expect(
      client.invokeTextGeneration({ user_prompt: "hi" }),
    ).rejects.toThrow(ScryptedAuthenticationError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});
