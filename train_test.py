from rfdetr import RFDETRNano

def main():
    model = RFDETRNano()
    model.train(
        dataset_dir="basketball-player-detection-2-13",
        epochs=10,
        batch_size=4,
        grad_accum_steps=2
    )

if __name__ == "__main__":
    main()
