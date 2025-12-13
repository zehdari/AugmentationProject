from rfdetr import RFDETRNano

def main():
    model = RFDETRNano()
    model.train(
        dataset_dir="reindexcoco",
        epochs=1,
        batch_size=12,
        grad_accum_steps=1
    )

if __name__ == "__main__":
    main()
