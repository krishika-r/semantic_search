"""The script for training/inferencing a question answering
"""

import pandas as pd
from transformers import pipeline
import subprocess
from subprocess import CalledProcessError
import evaluate
import os
import json
from pandas import json_normalize
from typing import Optional
from helper import generate_jsonl
import pathlib

class QnA:
    def __init__(self):
        self.max_answer_length = 50
        self.per_device_train_batch_size = 8 
        self.learning_rate = 3e-5 
        self.num_train_epochs = 2 
        self.max_seq_length = 384 
        self.doc_stride = 128 
        self.n_best_size = 20
        
    def train(self, train_data_path : str, output_path : str, model_name: Optional[str]=None, valn_path :Optional[str]=None, **kwargs):
        self.train_data_path = train_data_path
        self.valn_path = valn_path
        self.output_path = output_path
        self.model_name = model_name
        
        if not model_name:
            self.model_name = "deepset/tinyroberta-squad2"
            print("No model name specified. Considering default model : deepset/tinyroberta-squad2")

        if not train_data_path or not output_path:
            raise NameError("Provide train_data_path/output_path in the input")
        
        if 'max_answer_length' in kwargs:
            self.max_answer_length = kwargs.get('max_answer_length')
        if 'per_device_train_batch_size' in kwargs:
            self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
        if 'num_train_epochs' in kwargs:
            self.num_train_epochs = kwargs.get('num_train_epochs')
        if 'max_seq_length' in kwargs:
            self.max_seq_length = kwargs.get('max_seq_length')
        if 'doc_stride' in kwargs:
            self.doc_stride = kwargs.get('doc_stride')
        if 'n_best_size' in kwargs:
            self.n_best_size = kwargs.get('n_best_size')
        
        print("Executed")
        print('Model Name: {}'.format(self.model_name))

        model_params = [
            "python",
            "run_qa.py",
            "--model_name_or_path",
            self.model_name,
            "--train_file",
            self.train_data_path,
            "--do_train",
            f"--max_answer_length={self.max_answer_length}",
            "--version_2_with_negative",
            f"--per_device_train_batch_size={self.per_device_train_batch_size}",
            f"--learning_rate={self.learning_rate}",
            f"--num_train_epochs={self.num_train_epochs}",
            f"--max_seq_length={self.max_seq_length}",
            f"--doc_stride={self.doc_stride}",
            f"--n_best_size={self.n_best_size}",
        ]
        # command line argument for validation
        validation_cli = [
            "--do_eval",
            "--validation_file",
            self.valn_path,
            "--output_dir",
            self.output_path,
            "--overwrite_output_dir", # to overwrite the existing files
        ]

        # command line argument for train
        train_cli = [
            "--output_dir",
            self.output_path,
            "--overwrite_output_dir", # to overwrite the existing files
        ]
 
        if self.train_data_path and self.valn_path:
            model_params += validation_cli
            print("validation")
            print("training started")
            # Start the training
            try: 
                trainer = subprocess.run(
                    model_params,
                    check=True,
                    capture_output=True,
                )
            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))

            # saving jsonl output for validation file
            generate_jsonl(
                self.valn_path,
                self.output_path,
                self.model_name,
                "validation",
            )

        else:
            model_params += train_cli
            print("train")
            print("training started")
            # Start the training
            try: 
                trainer = subprocess.run(
                    model_params,
                    check=True,
                    capture_output=True,
                )
            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))

        print("Trainer output: ",trainer.stdout)

        if self.train_data_path:
            print("train prediction")
          #train prediction
            self.predict(
                model_name=self.model_name,
                output_path=os.path.join(self.output_path,"train_prediction"),
                test_path=self.train_data_path,
                is_train=True,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                max_answer_length=self.max_answer_length,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                doc_stride=self.doc_stride,
                max_seq_length=self.max_seq_length,
                n_best_size=self.n_best_size,
            )
        
        return


    def predict(
        self,
        question : str=None,
        context : str=None,
        model_name : Optional[str]=None,
        output_path : str =None,
        test_path: str = None,
        is_train=False, 
        **kwargs):
        """
        Inference function

        Parameters:
        model_name_or_path (str):model name (extractive qna)
        question,context (str) or dataframe

        Returns:
        dict: predicted answer
        
        """
        if not model_name:
            model_name = "deepset/tinyroberta-squad2"
            print("No model name specified. Considering default model : deepset/tinyroberta-squad2")
      
        print("Executed")
        print('Model Name: {}'.format(model_name))

        if isinstance(context,str) and isinstance(question,str):        
            question_answerer = pipeline('question-answering', model=model_name)
            res = question_answerer(question=question,context=context,**kwargs)
            df = pd.DataFrame([res])
            df['context'] = context
            df['question'] = question
            df['model_name'] = model_name
            df.rename(columns = {'answer':'predicted_answer'}, inplace = True)
            df= df[['question','context','predicted_answer','model_name','score','start','end']]
            # dict
            dictionary = df.to_dict(orient="records")
            return dictionary

        else:
            if not test_path or not output_path:
                raise TypeError(
                    "Please enter the test path/output path"
                )
            self.test_path = test_path
            self.output_path = output_path
            self.model_name = model_name

            if 'max_answer_length' in kwargs:
                self.max_answer_length = kwargs.get('max_answer_length')
            if 'per_device_train_batch_size' in kwargs:
                self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
            if 'learning_rate' in kwargs:
                self.learning_rate = kwargs.get('learning_rate')
            if 'num_train_epochs' in kwargs:
                self.num_train_epochs = kwargs.get('num_train_epochs')
            if 'max_seq_length' in kwargs:
                self.max_seq_length = kwargs.get('max_seq_length')
            if 'doc_stride' in kwargs:
                self.doc_stride = kwargs.get('doc_stride')
            if 'n_best_size' in kwargs:
                self.n_best_size = kwargs.get('n_best_size')

            model_params = [
                "python",
                "run_qa.py",
                "--model_name_or_path",
                self.model_name,
                "--test_file",
                self.test_path,
                 "--do_predict",
                f"--max_answer_length={self.max_answer_length}",
                f"--per_device_train_batch_size={self.per_device_train_batch_size}",
                f"--learning_rate={self.learning_rate}",
                f"--num_train_epochs={self.num_train_epochs}",
                f"--max_seq_length={self.max_seq_length}",
                f"--doc_stride={self.doc_stride}",
                f"--n_best_size={self.n_best_size}",
                "--overwrite_output_dir",  # to overwrite the existing files
                "--output_dir",
                self.output_path,
            ]

            # Start the training
            try:
                evaluator = subprocess.run(
                    model_params,
                    # shell=True
                    check=True,
                    capture_output=True,
                )
            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))

            generate_jsonl(
                self.test_path,
                self.output_path,
                self.model_name,
                "train" if is_train else "test",
            )

            print("Evaluation output: ",evaluator.stdout)