/**
 * Store for message extras (sources, verification) that the assistant-ui runtime doesn't preserve.
 * This allows AssistantMessageExtras to access this data for past chat messages.
 */

type MessageExtras = {
  sources?: Array<{
    title: string;
    url?: string;
    score?: number;
    chunk_id?: string;
  }>;
  verification?: {
    grounding_ratio?: number;
    num_claims?: number;
    num_grounded?: number;
    unsupported_claims?: string[];
  };
};

const messageExtrasMap = new Map<string, MessageExtras>();

export function setMessageExtras(messageId: string, extras: MessageExtras) {
  messageExtrasMap.set(messageId, extras);
}

export function getMessageExtras(messageId: string): MessageExtras | undefined {
  return messageExtrasMap.get(messageId);
}

export function clearMessageExtras() {
  messageExtrasMap.clear();
}
