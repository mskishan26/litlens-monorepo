"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/lib/auth-context";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  AssistantIf,
  ThreadListItemMorePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
} from "@assistant-ui/react";
import {
  ArchiveIcon,
  CheckIcon,
  MoreHorizontalIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";
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
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [renamingChatId, setRenamingChatId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteConfirmChatId, setDeleteConfirmChatId] = useState<string | null>(null);
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

      // Dispatch title for the currently selected chat (if any)
      if (chatId) {
        const matching = data.chats?.find((chat) => chat.ChatId === chatId);
        if (matching) {
          window.dispatchEvent(
            new CustomEvent("chat-title-loaded", {
              detail: { chatId, title: matching.title?.trim() || "Chat" },
            }),
          );
        }
      }
    } finally {
      setIsLoading(false);
    }
  }, [user?.uid, chatId]);

  useEffect(() => {
    void loadChats();
  }, [loadChats]);

  const handleSelectChat = useCallback(
    (id: string) => {
      // Dispatch title immediately so header updates without waiting for API
      const chat = chats.find((c) => c.ChatId === id);
      if (chat) {
        window.dispatchEvent(
          new CustomEvent("chat-title-loaded", {
            detail: { chatId: id, title: chat.title?.trim() || "Chat" },
          }),
        );
      }

      const params = new URLSearchParams(searchParams.toString());
      params.set("chatId", id);
      router.push(`${pathname}?${params.toString()}`);
    },
    [chats, pathname, router, searchParams],
  );

  const handleRenameChat = useCallback(
    async (id: string, nextTitle: string) => {
      if (!user?.uid) return;
      const trimmedTitle = nextTitle.trim();
      if (!trimmedTitle) return;

      setActiveChatId(id);
      try {
        const response = await fetch(`/api/chats/${id}`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            "x-user-id": user.uid,
            "x-user-anonymous": "false",
          },
          body: JSON.stringify({ title: trimmedTitle }),
        });

        if (!response.ok) {
          console.warn("Failed to rename chat", { chatId: id, status: response.status });
          return;
        }

        setChats((prev) =>
          prev.map((chat) =>
            chat.ChatId === id ? { ...chat, title: trimmedTitle } : chat,
          ),
        );
        window.dispatchEvent(
          new CustomEvent("chat-title-updated", {
            detail: { chatId: id, title: trimmedTitle },
          }),
        );
        setRenamingChatId(null);
        setRenameValue("");
      } finally {
        setActiveChatId(null);
      }
    },
    [user?.uid],
  );

  const handleDeleteChat = useCallback(
    async (id: string) => {
      if (!user?.uid) return;
      setActiveChatId(id);
      try {
        const response = await fetch(`/api/chats/${id}`, {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
            "x-user-id": user.uid,
            "x-user-anonymous": "false",
          },
        });

        if (!response.ok) {
          console.warn("Failed to delete chat", { chatId: id, status: response.status });
          return;
        }

        setChats((prev) => prev.filter((chat) => chat.ChatId !== id));

        if (chatId === id) {
          router.push(pathname);
        }
      } finally {
        setActiveChatId(null);
      }
    },
    [chatId, pathname, router, user?.uid],
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
            <div
              key={chat.id}
              className={`aui-thread-list-item group/chat flex h-9 items-center gap-2 rounded-lg px-3 text-start text-sm transition-colors hover:bg-muted focus-visible:bg-muted focus-visible:outline-none ${chatId === chat.id ? "bg-muted" : ""
                }`}
            >
              <div className="flex min-w-0 flex-1 items-center gap-2">
                {renamingChatId === chat.id ? (
                  <div className="flex w-full items-center gap-2">
                    <Input
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      onClick={(event) => event.stopPropagation()}
                      onBlur={() => {
                        // Cancel rename on blur (clicking elsewhere) - just restore original name
                        setRenamingChatId(null);
                        setRenameValue("");
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          const nextValue = renameValue;
                          setRenamingChatId(null);
                          setRenameValue("");
                          void handleRenameChat(chat.id, nextValue);
                        }
                        if (event.key === "Escape") {
                          event.preventDefault();
                          setRenamingChatId(null);
                          setRenameValue("");
                        }
                      }}
                      className="h-7"
                      autoFocus
                      aria-label="Rename chat"
                    />
                  </div>
                ) : (
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2 text-start"
                    onClick={() => handleSelectChat(chat.id)}
                  >
                    <span className="truncate">{chat.title}</span>
                  </button>
                )}
              </div>
              {deleteConfirmChatId === chat.id ? (
                <div className="flex items-center gap-1">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 text-destructive/80 hover:text-destructive"
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleDeleteChat(chat.id);
                      setDeleteConfirmChatId(null);
                    }}
                    disabled={activeChatId === chat.id}
                  >
                    Delete
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7"
                    onClick={(event) => {
                      event.stopPropagation();
                      setDeleteConfirmChatId(null);
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              ) : (
                renamingChatId === chat.id ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onMouseDown={(event) => {
                      // Prevent blur from firing before click
                      event.preventDefault();
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      const nextValue = renameValue;
                      setRenamingChatId(null);
                      setRenameValue("");
                      void handleRenameChat(chat.id, nextValue);
                    }}
                    disabled={activeChatId === chat.id}
                  >
                    <CheckIcon className="size-3.5" />
                    <span className="sr-only">Save rename</span>
                  </Button>
                ) : (
                  <ThreadListItemMorePrimitive.Root>
                    <ThreadListItemMorePrimitive.Trigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 opacity-0 transition-opacity group-hover/chat:opacity-100 data-[state=open]:bg-accent data-[state=open]:opacity-100"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <MoreHorizontalIcon className="size-3.5" />
                        <span className="sr-only">More options</span>
                      </Button>
                    </ThreadListItemMorePrimitive.Trigger>
                    <ThreadListItemMorePrimitive.Content
                      side="bottom"
                      align="end"
                      className="z-50 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <ThreadListItemMorePrimitive.Item
                        className="flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        onClick={(event) => {
                          event.stopPropagation();
                          setDeleteConfirmChatId(null);
                          setRenamingChatId(chat.id);
                          setRenameValue(chat.title);
                        }}
                        aria-disabled={activeChatId === chat.id}
                      >
                        <PencilIcon className="size-4" />
                        Rename
                      </ThreadListItemMorePrimitive.Item>
                      <ThreadListItemMorePrimitive.Item
                        className="flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-destructive/80 outline-none hover:bg-destructive/15 hover:text-destructive focus:bg-destructive/15"
                        onClick={(event) => {
                          event.stopPropagation();
                          setRenamingChatId(null);
                          setRenameValue("");
                          setDeleteConfirmChatId(chat.id);
                        }}
                        aria-disabled={activeChatId === chat.id}
                      >
                        <Trash2Icon className="size-4" />
                        Delete
                      </ThreadListItemMorePrimitive.Item>
                    </ThreadListItemMorePrimitive.Content>
                  </ThreadListItemMorePrimitive.Root>
                )
              )}
            </div>
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
  const router = useRouter();
  const pathname = usePathname();

  return (
    <Button
      variant="outline"
      className="aui-thread-list-new h-9 justify-start gap-2 rounded-lg px-3 text-sm hover:bg-muted data-active:bg-muted"
      onClick={() => router.push(pathname)}
    >
      <PlusIcon className="size-4" />
      New Thread
    </Button>
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
