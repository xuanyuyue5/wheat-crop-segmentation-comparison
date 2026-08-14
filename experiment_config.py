DATA_PATHS = {
    "train": "data/train",
    "validation": "data/validation",
    "test": "data/test",
}


RESULTS_DIR = "results"


EXPERIMENTS_BY_MODEL = [
    {
        "model_title": "U-Net",
        "model_key": "unet",
        "method_class": "UNetMethod",
        "sections": [
            {
                "section_title": "Main Experiment",
                "section_key": "main_experiment",
                "runs": [
                    {
                        "display_name": "U-Net",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "unet_main",
                    }
                ],
            },
            {
                "section_title": "Ablation: Pretraining",
                "section_key": "ablation_pretraining",
                "runs": [
                    {
                        "display_name": "U-Net (Pretrained)",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "unet_pretrained",
                    },
                    {
                        "display_name": "U-Net (Scratch)",
                        "kwargs": {"pretrained": False, "use_augmentation": True},
                        "tag": "unet_scratch",
                    },
                ],
            },
            {
                "section_title": "Ablation: Augmentation",
                "section_key": "ablation_augmentation",
                "runs": [
                    {
                        "display_name": "U-Net (With Augmentation)",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "unet_with_aug",
                    },
                    {
                        "display_name": "U-Net (Without Augmentation)",
                        "kwargs": {"pretrained": True, "use_augmentation": False},
                        "tag": "unet_without_aug",
                    },
                ],
            },
        ],
    },
    {
        "model_title": "Attention U-Net",
        "model_key": "attention_unet",
        "method_class": "AttentionUNetMethod",
        "sections": [
            {
                "section_title": "Main Experiment",
                "section_key": "main_experiment",
                "runs": [
                    {
                        "display_name": "Attention U-Net",
                        "kwargs": {"use_augmentation": True},
                        "tag": "attention_unet_main",
                    }
                ],
            },
            {
                "section_title": "Ablation: Augmentation",
                "section_key": "ablation_augmentation",
                "runs": [
                    {
                        "display_name": "Attention U-Net (With Augmentation)",
                        "kwargs": {"use_augmentation": True},
                        "tag": "attention_unet_with_aug",
                    },
                    {
                        "display_name": "Attention U-Net (Without Augmentation)",
                        "kwargs": {"use_augmentation": False},
                        "tag": "attention_unet_without_aug",
                    },
                ],
            },
            {
                "section_title": "Ablation: Attention Module",
                "section_key": "ablation_attention_module",
                "runs": [
                    {
                        "display_name": "U-Net (Baseline)",
                        "method_class": "UNetMethod",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "unet_baseline",
                    },
                    {
                        "display_name": "Attention U-Net",
                        "method_class": "AttentionUNetMethod",
                        "kwargs": {"use_augmentation": True},
                        "tag": "attention_unet_attention_module",
                    }
                ],
            },
        ],
    },
    {
        "model_title": "DeepLabV3+",
        "model_key": "deeplabv3plus",
        "method_class": "DLv3Plus",
        "sections": [
            {
                "section_title": "Main Experiment",
                "section_key": "main_experiment",
                "runs": [
                    {
                        "display_name": "DeepLabV3+",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "deeplabv3plus_main",
                    }
                ],
            },
            {
                "section_title": "Ablation: Pretraining",
                "section_key": "ablation_pretraining",
                "runs": [
                    {
                        "display_name": "DeepLabV3+ (Pretrained)",
                        "kwargs": {"pretrained": True, "use_augmentation": True},
                        "tag": "deeplabv3plus_pretrained",
                    },
                    {
                        "display_name": "DeepLabV3+ (Scratch)",
                        "kwargs": {"pretrained": False, "use_augmentation": True},
                        "tag": "deeplabv3plus_scratch",
                    },
                ],
            },
        ],
    },
]
