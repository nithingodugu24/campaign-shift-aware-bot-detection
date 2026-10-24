from src.synthetic import generate

if __name__ == "__main__":
    generate("data/processed/synthetic_source", seed=17, groups=48, accounts_per_group=12)
    generate("data/processed/synthetic_target", seed=913, groups=24, accounts_per_group=12, target=True)
    print("Prepared explicitly synthetic source and target fixtures")
