batch_size = 1024
valid = False
use_gpu = False
num_workers = 8
date = None
quantiles_ratio = [0.5]


field_config = dict(
    datetime="dt",
    category=[
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "day_of_week",
        "holiday_flag",
        "activity_flag",
    ],
    need_fill_na=[],
    need_encode_na=[],
)

dataset_config = dict(
    time_idx="time_idx",
    min_prediction_length=7,
    max_prediction_length=7,
    min_encoder_length=35,
    max_encoder_length=35,
    group_ids=["store_id", "product_id"],
    target=["sale_amount"],
    # weight="",
    static_categoricals=[
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ],
    time_varying_known_categoricals=[
        "day_of_week",
        "holiday_flag",
        "activity_flag",
    ],
    time_varying_known_reals=[
        "time_idx",
        "discount",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["sale_amount"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

model_config = dict(
    # learning_rate=0.0005,
    learning_rate = [1e-6, 1e-5, 1e-4, 5e-4, 5e-4, 5e-4, 1e-4],
    optimizer='adamw', # adamw, ranger
    dropout=0.15,
    weight_decay=1e-4,
    hidden_size=128,
    attention_head_size=8,
    hidden_continuous_size=16,
    reduce_on_plateau_patience=2,
    reduce_on_plateau_reduction=4,
    lstm_layers=1,
)

trainer_config = dict(
    max_epochs=7,
    enable_model_summary=True,
    # gradient_clip_val=0.1,
    gradient_clip_val=0.05,
    accumulate_grad_batches=2,
    default_root_dir=None,
    limit_train_batches=0.1
)
