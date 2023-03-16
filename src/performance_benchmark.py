import time
import numpy as np
from sklearn.metrics import accuracy_score

class PerformanceBenchmark:
    def __init__(self, model, tokenizer, dataset, optim_type=""):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self, true_labels, predicted_labels):
        return accuracy_score(true_labels, predicted_labels)

    def compute_size(self):
        num_parameters = sum(np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables)
        return {'model_size': num_parameters}

    def time_pipeline(self, data, warmup_runs=5):
        # Warm-up phase
        for _ in range(warmup_runs):
            _ = self.model(data, training=False)

        # Timed run
        start_time = time.time()
        _ = self.model(data, training=False)
        end_time = time.time()

        return {'time_taken': end_time - start_time}

    def run_benchmark(self):
        metrics = {}

        true_labels, predicted_labels = self.__get_predictions(self.model, self.dataset)
        accuracy = self.compute_accuracy(true_labels, predicted_labels)
        model_size = self.compute_size()
        data = next(iter(self.dataset.take(1)))
        processing_time = self.time_pipeline(data)

        metrics[self.optim_type] = {
            'accuracy': accuracy,
            **model_size,
            **processing_time
        }

        return metrics
    
    # TODO: this function is not belong to this class, modify later

    def __get_predictions(self, model, dataset):
        all_labels = []
        all_predictions = []
        
        for batch in dataset:
            inputs = batch[0]
            labels = batch[1].numpy()
            predictions = np.argmax(model(inputs, training=False).logits, axis=1)
            
            all_labels.extend(labels)
            all_predictions.extend(predictions)
        
        return np.array(all_labels), np.array(all_predictions)