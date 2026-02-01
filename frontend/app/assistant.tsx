"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  useChatRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { generateId } from "ai";
import { Thread } from "@/components/assistant-ui/thread";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { useAuth } from "@/lib/auth-context";
import { setMessageExtras, clearMessageExtras } from "@/lib/message-extras-store";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

export const Assistant = () => {
  const { user } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const rawChatId = searchParams.get("chatId");
  const urlChatId =
    rawChatId && rawChatId !== "undefined" && rawChatId !== "null"
      ? rawChatId
      : null;
  const chatIdsRef = useRef(new Map<string, string>());
  const prevChatIdRef = useRef<string | null>(null);
  const [chatSessionKey, setChatSessionKey] = useState(0);
  const [initialMessages, setInitialMessages] = useState<any[] | null>(null);

  useEffect(() => {
    if (!urlChatId) {
      chatIdsRef.current.clear();
      clearMessageExtras();
    }
  }, [urlChatId]);

  useEffect(() => {
    if (!urlChatId && prevChatIdRef.current !== null) {
      setChatSessionKey((key) => key + 1);
    }
    prevChatIdRef.current = urlChatId;
  }, [urlChatId]);



  useEffect(() => {
    let isActive = true;

    const loadMessages = async () => {
      // If we are switching to a chat we just created in this session, don't clear the messages
      // This prevents the "disappearing message" issue
      const isOptimisticChat =
        urlChatId &&
        (chatIdsRef.current.get(urlChatId) === urlChatId ||
          Array.from(chatIdsRef.current.values()).includes(urlChatId));

      if (!urlChatId) {
        if (isActive) setInitialMessages([]);
        return;
      }

      // Only clear messages if it's not a chat we just created locally
      if (isActive && !isOptimisticChat) setInitialMessages(null);

      try {
        const res = await fetch(`/api/chats/${urlChatId}/messages`, {
          headers: {
            ...(user?.uid ? { "x-user-id": user.uid } : {}),
            "x-user-anonymous": user?.uid ? "false" : "true",
          },
        });

        if (!isActive) return;

        if (!res.ok) {
          console.warn("Failed to load messages", {
            chatId: urlChatId,
            status: res.status,
          });
          setInitialMessages([]);
          return;
        }

        const data = (await res.json()) as {
          chat_id?: string;
          messages?: Array<{
            MessageId: string;
            query?: string;
            answer?: string;
            timestamp?: string;
            [key: string]: any;
          }>;
        };

        if (!isActive) return;

        const hydrated = (data.messages ?? []).flatMap((m) => {
          const ts = m.timestamp ? new Date(m.timestamp) : new Date();
          const items: any[] = [];
          const rawUserText =
            m.query ??
            m.Query ??
            m.question ??
            m.prompt ??
            m.input ??
            "";
          const rawAssistantText =
            m.answer ??
            m.Answer ??
            m.response ??
            m.output ??
            "";
          const userText =
            typeof rawUserText === "string"
              ? rawUserText
              : rawUserText?.text ?? rawUserText?.content ?? "";
          const assistantText =
            typeof rawAssistantText === "string"
              ? rawAssistantText
              : rawAssistantText?.text ?? rawAssistantText?.content ?? "";

          if (String(userText).trim()) {
            items.push({
              id: `${m.MessageId}-user`,
              role: "user" as const,
              content: userText,
              parts: [{ type: "text", text: userText }],
              createdAt: ts,
            });
          }

          if (String(assistantText).trim()) {
            const assistantMsgId = `${m.MessageId}-assistant`;

            // Store sources and verification in the extras store (read by AssistantMessageExtras)
            const extras: Record<string, any> = {};

            if (Array.isArray(m.sources) && m.sources.length > 0) {
              extras.sources = m.sources.map((source: any) => ({
                title: source.title ?? source.file_path ?? "Source",
                url: source.file_path,
                score: source.score,
                chunk_id: source.chunk_id,
              }));
            }

            if (m.hallucination) {
              extras.verification = {
                grounding_ratio: m.hallucination.grounding_ratio,
                num_claims: m.hallucination.num_claims,
                num_grounded: m.hallucination.num_grounded,
                unsupported_claims: m.hallucination.unsupported_claims,
              };
            }

            if (extras.sources || extras.verification) {
              setMessageExtras(assistantMsgId, extras);
            }

            items.push({
              id: assistantMsgId,
              role: "assistant" as const,
              content: assistantText,
              parts: [{ type: "text", text: assistantText }],
              createdAt: ts,
              status: { type: "complete" },
            });
          }

          if (!String(userText).trim()) {
            console.warn("Missing user message text", {
              messageId: m.MessageId,
              availableFields: Object.keys(m ?? {}),
            });
          }

          return items;
        });

        if (isActive) {
          setInitialMessages(hydrated);
        }
      } catch (error) {
        if (isActive) {
          console.error("Error loading messages", { chatId: urlChatId, error });
          setInitialMessages([]);
        }
      }
    };

    void loadMessages();

    return () => {
      isActive = false;
    };
  }, [urlChatId, user?.uid]);

  if (urlChatId && initialMessages === null) {
    return (
      <div className="flex h-dvh w-full items-center justify-center bg-background text-muted-foreground">
        Loading conversation...
      </div>
    );
  }

  return (
    <AssistantContent
      key={`session-${chatSessionKey}`}
      initialMessages={urlChatId && initialMessages ? initialMessages : []}
      urlChatId={urlChatId}
      chatIdsRef={chatIdsRef}
    />
  );
};

const AssistantContent = ({
  initialMessages,
  urlChatId,
  chatIdsRef,
}: {
  initialMessages: any[];
  urlChatId: string | null;
  chatIdsRef: React.MutableRefObject<Map<string, string>>;
}) => {
  const { user } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [messages, setMessages] = useState(initialMessages);
  const [chatTitle, setChatTitle] = useState<string>("New chat");

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    let isActive = true;

    const loadChatTitle = async () => {
      if (!urlChatId) {
        if (isActive) setChatTitle("New chat");
        return;
      }

      try {
        const response = await fetch("/api/chats", {
          headers: {
            ...(user?.uid ? { "x-user-id": user.uid } : {}),
            "x-user-anonymous": user?.uid ? "false" : "true",
          },
        });

        if (!isActive) return;

        if (!response.ok) {
          setChatTitle("Chat");
          return;
        }

        const data = (await response.json()) as {
          chats?: Array<{ ChatId: string; title?: string }>;
        };
        const matching = data.chats?.find((chat) => chat.ChatId === urlChatId);
        const title = matching?.title?.trim();
        setChatTitle(title ? title : "Chat");
      } catch (error) {
        if (isActive) {
          console.warn("Failed to load chat title", { chatId: urlChatId, error });
          setChatTitle("Chat");
        }
      }
    };

    void loadChatTitle();

    const handleTitleUpdated = (event: Event) => {
      if (!urlChatId) return;
      const detail = (event as CustomEvent<{ chatId: string; title: string }>).detail;
      if (detail?.chatId === urlChatId) {
        setChatTitle(detail.title?.trim() ? detail.title : "Chat");
      }
    };

    window.addEventListener("chat-title-updated", handleTitleUpdated);

    return () => {
      isActive = false;
      window.removeEventListener("chat-title-updated", handleTitleUpdated);
    };
  }, [urlChatId, user?.uid]);

  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: "/api/chat",
        prepareSendMessagesRequest: async (options) => {
          const threadId = options.id ?? "default";
          const existingChatId = chatIdsRef.current.get(threadId);
          const chatId = existingChatId ?? urlChatId ?? generateId();

          if (!existingChatId) {
            chatIdsRef.current.set(threadId, chatId);
          }

          if (!urlChatId || urlChatId !== chatId) {
            const params = new URLSearchParams(searchParams.toString());
            params.set("chatId", chatId);
            router.replace(`${pathname}?${params.toString()}`);
          }

          options.body = {
            ...(options.body ?? {}),
            chatId,
            conversation_id: chatId,
          };

          return {
            api: options.api,
            headers: {
              ...(options.headers ?? {}),
              ...(user?.uid ? { "x-user-id": user.uid } : {}),
              "x-user-anonymous": user?.uid ? "false" : "true",
            },
            credentials: options.credentials,
            body: {
              ...(options.body ?? {}),
              id: options.id,
              messages: options.messages,
              trigger: options.trigger,
              messageId: options.messageId,
              metadata: options.requestMetadata,
            },
          };
        },
      }),
    [pathname, router, searchParams, urlChatId, user?.uid],
  );
  const runtime = useChatRuntime({
    transport,
    messages,
    // @ts-ignore
    onMessagesChange: setMessages,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider>
        <div className="flex h-dvh w-full pr-0.5">
          <ThreadListSidebar />
          <SidebarInset>
            <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger />
              <Separator orientation="vertical" className="mr-2 h-4" />
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbPage>{chatTitle}</BreadcrumbPage>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
            </header>
            <div className="flex-1 overflow-hidden">
              <Thread />
            </div>
          </SidebarInset>
        </div>
      </SidebarProvider>
    </AssistantRuntimeProvider>
  );
};
