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
  const [initialMessages, setInitialMessages] = useState<any[] | null>(null);



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
            items.push({
              id: `${m.MessageId}-assistant`,
              role: "assistant" as const,
              content: assistantText,
              parts: [{ type: "text", text: assistantText }],
              createdAt: ts,
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
      initialMessages={initialMessages ?? []}
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

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

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
                  <BreadcrumbItem className="hidden md:block">
                    <BreadcrumbLink
                      href="https://www.assistant-ui.com/docs/getting-started"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Build Your Own ChatGPT UX
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator className="hidden md:block" />
                  <BreadcrumbItem>
                    <BreadcrumbPage>Starter Template</BreadcrumbPage>
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
