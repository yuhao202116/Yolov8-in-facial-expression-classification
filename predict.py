from ultralytics import YOLO

def predict(frame):
  # Load your custom trained model
  model = YOLO("./runs/classify/train5(minset_30epoch)/weights/best.pt")

  # Predict with the model
  results = model(frame)

  # Print results
  for result in results:
    print(result.boxes)  # Print detection boxes
    result.show()  # Display the annotated image
