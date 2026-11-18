import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n";
import { MessageBubble } from "../MessageBubble";
import type { StoredAgentMessage } from "@/stores/agent";

vi.mock("react-markdown", () => ({
  default: ({ children }: { children: string }) => <div data-testid="markdown">{children}</div>,
}));
vi.mock("remark-gfm", () => ({ default: () => {} }));
vi.mock("../RunCompleteCard", () => ({ RunCompleteCard: () => null }));

const userMsg: StoredAgentMessage = {
  id: "u-1",
  type: "user",
  content: "Top 5 crypto gainers",
  timestamp: 0,
};

describe("MessageBubble user message editing", () => {
  beforeAll(async () => {
    await i18n.changeLanguage("en");
  });

  it("hides the edit button when no edit handler is given", () => {
    render(<MessageBubble msg={userMsg} />);
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull();
  });

  it("sends the edited text and leaves edit mode", async () => {
    const onEdit = vi.fn();
    render(<MessageBubble msg={userMsg} onEdit={onEdit} />);

    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    const box = screen.getByRole("textbox", { name: "Edit" });
    expect(box).toHaveValue("Top 5 crypto gainers");

    await userEvent.clear(box);
    await userEvent.type(box, "  Top 10 crypto gainers  ");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(onEdit).toHaveBeenCalledWith(userMsg, "Top 10 crypto gainers");
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("cancels with Escape without sending", async () => {
    const onEdit = vi.fn();
    render(<MessageBubble msg={userMsg} onEdit={onEdit} />);

    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    await userEvent.type(screen.getByRole("textbox", { name: "Edit" }), " more{Escape}");

    expect(onEdit).not.toHaveBeenCalled();
    expect(screen.getByText("Top 5 crypto gainers")).toBeInTheDocument();
  });

  it("does not send an empty edit", async () => {
    const onEdit = vi.fn();
    render(<MessageBubble msg={userMsg} onEdit={onEdit} />);

    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    await userEvent.clear(screen.getByRole("textbox", { name: "Edit" }));

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(onEdit).not.toHaveBeenCalled();
  });
});
