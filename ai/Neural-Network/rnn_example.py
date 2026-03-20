from __future__ import annotations

from _venv_bootstrap import rerun_with_nearest_venv

rerun_with_nearest_venv()

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CharSequenceDataset(Dataset):
    def __init__(self, encoded_text: list[int], sequence_length: int = 40, step: int = 3) -> None:
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        last_index = len(encoded_text) - sequence_length - 1
        for start in range(0, last_index, step):
            chunk = encoded_text[start : start + sequence_length + 1]
            self.inputs.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.rnn = nn.RNN(32, 128, batch_first=True)
        self.output = nn.Linear(128, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.output(out)
        return logits, hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small character-level RNN.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent / "rnn_corpus.txt",
    )
    parser.add_argument("--generate-length", type=int, default=120)
    return parser.parse_args()


def load_corpus(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip().lower()
    if len(text) < 200:
        raise ValueError("Corpus is too small for the example.")
    return text


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / batches


@torch.no_grad()
def generate_text(
    model: CharRNN,
    prefix: str,
    char_to_index: dict[str, int],
    index_to_char: dict[int, str],
    device: torch.device,
    length: int,
) -> str:
    safe_prefix = "".join(ch for ch in prefix if ch in char_to_index)
    if not safe_prefix:
        safe_prefix = next(iter(char_to_index))

    model.eval()
    input_ids = torch.tensor([[char_to_index[ch] for ch in safe_prefix]], device=device)
    _, hidden = model(input_ids)
    current = input_ids[:, -1:]
    generated = safe_prefix

    for _ in range(length):
        logits, hidden = model(current, hidden)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_char = index_to_char[int(next_id.item())]
        generated += next_char
        current = next_id

    return generated


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = torch.device(args.device)
    corpus = load_corpus(args.corpus)
    chars = sorted(set(corpus))
    char_to_index = {char: index for index, char in enumerate(chars)}
    index_to_char = {index: char for char, index in char_to_index.items()}
    encoded = [char_to_index[char] for char in corpus]

    dataset = CharSequenceDataset(encoded)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CharRNN(vocab_size=len(chars)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Device: {device}")
    print(f"Corpus length: {len(corpus)} characters")
    print(f"Vocabulary size: {len(chars)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, loss_fn, optimizer, device, len(chars))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f}")

    sample = generate_text(
        model=model,
        prefix=corpus[:24],
        char_to_index=char_to_index,
        index_to_char=index_to_char,
        device=device,
        length=args.generate_length,
    )
    print("\nGenerated sample:")
    print(sample)


if __name__ == "__main__":
    main()
