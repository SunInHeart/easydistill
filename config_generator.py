import argparse
import json
import os
# from datetime import datetime

def create_kd_config_from_cli():
    """从命令行参数创建知识蒸馏配置并生成JSON文件"""
    parser = argparse.ArgumentParser(description="创建知识蒸馏配置文件")
    
    # KD 类型 (黑盒 or 白盒)
    parser.add_argument("--kd_type", required=True, default="black-box", 
                        choices=["black-box", "white-box"], help="知识蒸馏的类型 (必需)")
    
    # 数据集参数
    parser.add_argument("--instruction_path", default="data/distil_qwen_100k.json",
                        help="指令数据集路径 (默认: data/distil_qwen_100k.json)")
    parser.add_argument("--labeled_path", default="data/labeled.json",
                        help="标签数据集路径 (默认: data/labeled.json)")
    parser.add_argument("--template_path", 
                        default="configs/chat_template/chat_template_kd.jinja",
                        help="模板路径 (默认: configs/chat_template/chat_template_kd.jinja)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")

    # 模型参数
    parser.add_argument("--teacher_model", required=True, help="教师模型路径 (必需)")
    parser.add_argument("--student_model", required=True, help="学生模型路径 (必需)")

    # 推理参数
    parser.add_argument("--temperature", type=float, default=0.8, help="温度 (默认: 0.8)")
    parser.add_argument("--max_model_len", type=int, default=4096, help="最大模型长度: (默认: 4096)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大新生成token数: (默认: 512)")

    # 蒸馏参数
    parser.add_argument("--kd_ratio", type=float, default=0.5, help="知识蒸馏比率 (默认: 0.5)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度 (默认: 512)")
    parser.add_argument("--distillation_type", default="forward_kld", help="蒸馏损失函数类型 (默认: forward_kld)")
    
    # 训练参数
    parser.add_argument("--output_dir", required=True, help="输出目录路径 (必需)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数 (默认: 3)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小 (默认: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="梯度累积步数 (默认: 8)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大长度 (默认: 512)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率 (默认: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减 (默认: 0.05)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例 (默认: 0.1)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                        help="学习率调度器类型: (默认: cosine)")
    
    # 日志和保存参数
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="保存步数 (默认: 1000)")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="日志记录步数 (默认: 1)")
    
    # 输出参数
    parser.add_argument("--config_output", 
                        default="test/kd_config.json",
                        help="配置文件输出路径 (默认: kd_config.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的配置文件")
    
    args = parser.parse_args()

    # 生成配置文件
    if args.kd_type == "black-box":
        # 构建配置字典
        config = {
            "job_type": "kd_black_box_local",
            "dataset": {
                "instruction_path": args.instruction_path,
                "labeled_path": args.labeled_path,
                "template": args.template_path,
                "seed": args.seed
            },
            "inference":{
                "enable_chunked_prefill": True,
                "seed": 777,
                "gpu_memory_utilization": 0.9,
                "temperature": args.temperature,
                "trust_remote_code": True,
                "enforce_eager": False,
                "max_model_len": args.max_model_len,
                "max_new_tokens": args.max_new_tokens
            },
            "models": {
                "teacher": args.teacher_model,
                "student": args.student_model
            },
            "training": {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_epochs,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "save_steps": args.save_steps,
                "logging_steps": args.logging_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type
            }
        }
    elif args.kd_type == "white-box":
        config = {
            "job_type": "kd_white_box",
            "dataset": {
                "instruction_path": args.instruction_path,
                "labeled_path": args.instruction_path,
                "logits_path": "data/logits.json",
                "template" : args.template_path,
                "seed": 42
            },
            "inference":{
                "enable_chunked_prefill": True,
                "seed": 777,
                "gpu_memory_utilization": 0.9,
                "temperature": args.temperature,
                "trust_remote_code": True,
                "enforce_eager": False,
                "max_model_len": args.max_model_len,
                "max_new_tokens": args.max_new_tokens,
                "top_logits_num": 10
            },
            "distillation": {
                "kd_ratio": args.kd_ratio,
                "max_seq_length": args.max_seq_length,
                "distillation_type": args.distillation_type
            },
            "models": {
                "teacher": args.teacher_model,
                "student": args.student_model
            },
            "training": {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_epochs,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "save_steps": args.save_steps,
                "logging_steps": args.logging_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type
            }
        }
    
    # 检查输出文件是否存在
    if os.path.exists(args.config_output) and not args.overwrite:
        print(f"错误: 配置文件 {args.config_output} 已存在，使用 --overwrite 来覆盖")
        return None
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.config_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入JSON文件
    try:
        with open(args.config_output, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置文件已生成: {args.config_output}")
        print(f"📁 教师模型: {args.teacher_model}")
        print(f"📁 学生模型: {args.student_model}")
        print(f"📂 输出目录: {args.output_dir}")
        print(f"📊 训练轮数: {args.num_epochs}")
        print(f"🎯 学习率: {args.learning_rate}")
        
        return config
        
    except Exception as e:
        print(f"❌ 写入配置文件失败: {e}")
        return None

if __name__ == "__main__":
    
    # 创建蒸馏配置文件
    config = create_kd_config_from_cli()

