from __future__ import annotations

from src.world import WorldModel


def main():
    w = WorldModel(frame_id=1)
    print("OK:", w.summary())


if __name__ == "__main__":
    main()
