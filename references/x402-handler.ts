/**
 * x402 payment handler for ScryptedAI SDK.
 *
 * Handles the complete x402 payment flow including invoice extraction
 * from 402 responses and payment verification using Base blockchain data.
 */

import { X402Invoice } from "./models";

export interface PaymentReceipt {
  signature: string;
  authorization: {
    from: string;
    to: string;
    value: string;
    validAfter: string;
    validBefore: string;
    nonce: string;
  };
  transactionHash?: string;
  blockNumber?: string;
  status?: string;
}

export interface PaymentPayload {
  x402Version: number;
  scheme: string;
  network: string;
  payload: {
    signature: string;
    authorization: {
      from: string;
      to: string;
      value: string;
      validAfter: string;
      validBefore: string;
      nonce: string;
    };
  };
}

/**
 * Handles x402 payment flow with Base blockchain integration.
 */
export class X402PaymentHandler {
  private client: any; // Reference to ScryptedClient

  constructor(client: any) {
    this.client = client;
  }

  /**
   * Extract x402 invoice from 402 Payment Required response.
   *
   * @param responseData - The 402 response data from API
   * @returns Tuple of (invoice_data, full_challenge_data)
   */
  async extractInvoiceFrom402(
    responseData: any
  ): Promise<[X402Invoice, any]> {
    // Extract the x402 challenge data
    // The API returns this in the 402 response
    const accepts = responseData.accepts || [];
    const invoice = accepts[0] || {};

    // Also preserve the full challenge data
    const challengeData = { ...responseData };

    return [invoice as X402Invoice, challengeData];
  }

  /**
   * Construct X-PAYMENT header from invoice and transaction hash.
   *
   * This:
   * 1. Fetches transaction from Base blockchain
   * 2. Extracts receipt data
   * 3. Builds PaymentPayload
   * 4. Base64 encodes it
   *
   * @param invoice - x402 invoice from 402 response
   * @param txHash - Blockchain transaction hash (0x...)
   * @returns Base64-encoded X-PAYMENT header value
   */
  async constructXPaymentHeader(
    invoice: X402Invoice,
    txHash: string
  ): Promise<string> {
    // Step 1: Fetch transaction from blockchain
    const txReceipt = await BaseChainReader.getTransactionReceipt(txHash);
    const txData = await BaseChainReader.getTransaction(txHash);

    // Step 2: Extract payment receipt
    const receipt = BaseChainReader.extractPaymentReceipt(
      txReceipt,
      txData,
      invoice
    );

    // Step 3: Build PaymentPayload
    const paymentPayload: PaymentPayload = {
      x402Version: 1,
      scheme: invoice.scheme || "exact",
      network: invoice.network || "base",
      payload: {
        signature: receipt.signature,
        authorization: receipt.authorization,
      },
    };

    // Step 4: Base64 encode (compact JSON to avoid extra spaces)
    const payloadJson = JSON.stringify(paymentPayload);
    const encoded = Buffer.from(payloadJson).toString("base64");

    return encoded;
  }

  /**
   * Validate that invoice has required x402 fields.
   *
   * @param invoice - Invoice data to validate
   * @returns True if valid
   */
  validateX402Invoice(invoice: X402Invoice): boolean {
    const requiredFields = [
      "scheme",
      "network",
      "maxAmountRequired",
      "payTo",
      "asset",
    ];

    return requiredFields.every((field) => field in invoice);
  }
}

/**
 * Blockchain reader for Base network using public RPC endpoints.
 *
 * Uses free public RPC endpoints to fetch transaction data without
 * requiring any credentials or API keys.
 */
class BaseChainReader {
  // Public Base RPC endpoints (fallback order)
  private static readonly BASE_RPC_ENDPOINTS = [
    "https://mainnet.base.org", // Official Base RPC
    "https://base-rpc.publicnode.com", // PublicNode
    "https://base.meowrpc.com", // MeowRPC
  ];

  /**
   * Fetch transaction receipt from Base blockchain.
   *
   * @param txHash - Transaction hash (0x...)
   * @returns Transaction receipt dict
   * @throws Error - If transaction not found on any RPC endpoint
   */
  static async getTransactionReceipt(txHash: string): Promise<any> {
    const rpcPayload = {
      jsonrpc: "2.0",
      id: 1,
      method: "eth_getTransactionReceipt",
      params: [txHash],
    };

    for (const endpoint of this.BASE_RPC_ENDPOINTS) {
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(rpcPayload),
        });

        if (!response.ok) continue;

        const data = await response.json();
        if (data.result) {
          return data.result;
        }
      } catch (error) {
        continue;
      }
    }

    throw new Error(
      `Could not fetch transaction ${txHash} from any RPC endpoint`
    );
  }

  /**
   * Fetch full transaction details from Base blockchain.
   *
   * @param txHash - Transaction hash (0x...)
   * @returns Transaction data dict
   */
  static async getTransaction(txHash: string): Promise<any> {
    const rpcPayload = {
      jsonrpc: "2.0",
      id: 1,
      method: "eth_getTransactionByHash",
      params: [txHash],
    };

    for (const endpoint of this.BASE_RPC_ENDPOINTS) {
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(rpcPayload),
        });

        if (!response.ok) continue;

        const data = await response.json();
        if (data.result) {
          return data.result;
        }
      } catch (error) {
        continue;
      }
    }

    throw new Error(
      `Could not fetch transaction ${txHash} from any RPC endpoint`
    );
  }

  /**
   * Extract payment receipt data from transaction for x402.
   *
   * @param txReceipt - Transaction receipt from blockchain
   * @param txData - Full transaction data
   * @param invoice - x402 invoice from 402 response (accepts[0])
   * @returns Receipt data for PaymentPayload
   */
  static extractPaymentReceipt(
    txReceipt: any,
    txData: any,
    invoice: X402Invoice
  ): PaymentReceipt {
    // Extract authorization from invoice
    // If invoice has authorization field, use it
    // Otherwise, construct from transaction data
    let authorization = (invoice as any).authorization;

    if (!authorization) {
      // Construct from transaction data
      authorization = {
        from: txData.from,
        to: invoice.payTo, // Use payTo from invoice
        value: invoice.maxAmountRequired || "0",
        validAfter: "0",
        validBefore: String(2 ** 64 - 1), // Max uint64
        nonce: txData.nonce || txReceipt.transactionIndex || "0",
      };
    }

    return {
      signature: txData.signature || "0x",
      authorization,
      transactionHash: txReceipt.transactionHash,
      blockNumber: txReceipt.blockNumber,
      status: txReceipt.status || "0x1",
    };
  }
}

