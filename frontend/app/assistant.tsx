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
import { useEffect, useMemo, useRef } from "react";

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

  useEffect(() => {
    if (urlChatId) {
      chatIdsRef.current.set("default", urlChatId);
      chatIdsRef.current.set(urlChatId, urlChatId);
    }
  }, [urlChatId]);
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
