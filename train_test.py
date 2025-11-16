from rfdetr import RFDETRNano

def main():
    model = RFDETRNano()
    model.train(
        dataset_dir="basketball-player-detection-2-13",
        epochs=5,
        batch_size=14,
        grad_accum_steps=1
    )

if __name__ == "__main__":
    main()
