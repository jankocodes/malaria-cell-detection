import argparse
import os
from data.util import *


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("img_id", type=int, help="Image ID to visualize")
    parser.add_argument(
        "split", choices=["train", "val"], help="Dataset split", default="train"
    )
    parser.add_argument(
        "--without_target",
        action="store_true",
        help="Whether to load and show targets",
    )
    parser.add_argument(
        "--rare_only",
        action="store_true",
        help="Whether to load and show only rare targets",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/vogelbacher23",
        help="Path to dataset",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Show letterbox-resized version instead of original",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save visualization to this path instead of displaying",
    )

    args = parser.parse_args()
    print(args)

    # Verify data exists
    assert os.path.exists(args.data_path), f"Data path not found: {args.data_path}"
    assert os.path.exists(
        os.path.join(args.data_path, args.split)
    ), f"Split '{args.split}' not found in {args.data_path}"

    visualize_sample(
        data_path=args.data_path,
        split=args.split,
        img_id=args.img_id,
        without_target=args.without_target,
        rare_only=args.rare_only,
        show_letterbox=args.letterbox,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
