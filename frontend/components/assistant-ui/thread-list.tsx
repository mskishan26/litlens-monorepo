"use client";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/lib/auth-context";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  AssistantIf,
  ThreadListItemMorePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
} from "@assistant-ui/react";
import { ArchiveIcon, MoreHorizontalIcon, PlusIcon } from "lucide-react";
import type { FC } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

type ChatSummary = {
  ChatId: string;
  title?: string;
  updated_at?: string;
  last_message_date?: string;
};

export const ThreadList: FC = () => {
  const { user } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatId = searchParams.get("chatId");

  const loadChats = useCallback(async () => {
    if (!user?.uid) {
      setChats([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("/api/chats", {
        headers: {
          "x-user-id": user.uid,
          "x-user-anonymous": "false",
        },
      });
      if (!response.ok) {
        setChats([]);
        return;
      }
      const data = (await response.json()) as { chats?: ChatSummary[] };
      setChats(data.chats ?? []);
    } finally {
      setIsLoading(false);
    }
  }, [user?.uid]);

  useEffect(() => {
    void loadChats();
  }, [loadChats]);

  const handleSelectChat = useCallback(
    (id: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("chatId", id);
      router.push(`${pathname}?${params.toString()}`);
    },
    [pathname, router, searchParams],
  );

  const chatItems = useMemo(
    () =>
      chats.map((chat) => ({
        id: chat.ChatId,
        title: chat.title?.trim() ? chat.title : "Title not found",
        updatedAt: chat.updated_at ?? chat.last_message_date,
      })),
    [chats],
  );

  return (
    <ThreadListPrimitive.Root className="aui-root aui-thread-list-root flex flex-col gap-1">
      <ThreadListNew />
      {isLoading ? <ThreadListSkeleton /> : null}
      {chatItems.length > 0 ? (
        <div className="flex flex-col gap-1">
          {chatItems.map((chat) => (
            <button
              key={chat.id}
              type="button"
              className={`aui-thread-list-item group flex h-9 items-center gap-2 rounded-lg px-3 text-start text-sm transition-colors hover:bg-muted focus-visible:bg-muted focus-visible:outline-none ${
                chatId === chat.id ? "bg-muted" : ""
              }`}
              onClick={() => handleSelectChat(chat.id)}
            >
              <span className="truncate">{chat.title}</span>
            </button>
          ))}
        </div>
      ) : (
        <AssistantIf condition={({ threads }) => threads.isLoading}>
          <ThreadListSkeleton />
        </AssistantIf>
      )}
    </ThreadListPrimitive.Root>
  );
};

const ThreadListNew: FC = () => {
  return (
    <ThreadListPrimitive.New asChild>
      <Button
        variant="outline"
        className="aui-thread-list-new h-9 justify-start gap-2 rounded-lg px-3 text-sm hover:bg-muted data-active:bg-muted"
      >
        <PlusIcon className="size-4" />
        New Thread
      </Button>
    </ThreadListPrimitive.New>
  );
};

const ThreadListSkeleton: FC = () => {
  return (
    <div className="flex flex-col gap-1">
      {Array.from({ length: 5 }, (_, i) => (
        <div
          key={i}
          role="status"
          aria-label="Loading threads"
          className="aui-thread-list-skeleton-wrapper flex h-9 items-center px-3"
        >
          <Skeleton className="aui-thread-list-skeleton h-4 w-full" />
        </div>
      ))}
    </div>
  );
};

const ThreadListItem: FC = () => {
  return (
    <ThreadListItemPrimitive.Root className="aui-thread-list-item group flex h-9 items-center gap-2 rounded-lg transition-colors hover:bg-muted focus-visible:bg-muted focus-visible:outline-none data-active:bg-muted">
      <ThreadListItemPrimitive.Trigger className="aui-thread-list-item-trigger flex h-full min-w-0 flex-1 items-center truncate px-3 text-start text-sm">
        <ThreadListItemPrimitive.Title fallback="New Chat" />
      </ThreadListItemPrimitive.Trigger>
      <ThreadListItemMore />
    </ThreadListItemPrimitive.Root>
  );
};

const ThreadListItemMore: FC = () => {
  return (
    <ThreadListItemMorePrimitive.Root>
      <ThreadListItemMorePrimitive.Trigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="aui-thread-list-item-more mr-2 size-7 p-0 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:bg-accent data-[state=open]:opacity-100 group-data-active:opacity-100"
        >
          <MoreHorizontalIcon className="size-4" />
          <span className="sr-only">More options</span>
        </Button>
      </ThreadListItemMorePrimitive.Trigger>
      <ThreadListItemMorePrimitive.Content
        side="bottom"
        align="start"
        className="aui-thread-list-item-more-content z-50 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
      >
        <ThreadListItemPrimitive.Archive asChild>
          <ThreadListItemMorePrimitive.Item className="aui-thread-list-item-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
            <ArchiveIcon className="size-4" />
            Archive
          </ThreadListItemMorePrimitive.Item>
        </ThreadListItemPrimitive.Archive>
      </ThreadListItemMorePrimitive.Content>
    </ThreadListItemMorePrimitive.Root>
  );
};
