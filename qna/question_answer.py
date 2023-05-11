"""The script for training and inferencing a question answering model
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
    """ QnA is an extractive question answering technique that extracts answer
        from the context. This is usually solved using BERT-like models.
        
    Examples
    --------
    >>> from question_answer import QnA
    >>> qna=QnA()
    """
    def __init__(self):
        self.max_answer_length = 50
        self.per_device_train_batch_size = 8 
        self.learning_rate = 3e-5 
        self.num_train_epochs = 2 
        self.max_seq_length = 384 
        self.doc_stride = 128 
        self.n_best_size = 20

        """QnA initialization

        Parameters
        ----------
        max_answer_length: The maximum length of an answer that can be generated. This is needed because the start 
                           and end predictions are not conditioned on one another.
        per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training
        learning_rate: The initial learning rate for ADAM
        num_train_epochs: Total number of training epochs to perform.
        max_seq_length: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,
                        sequences shorter will be padded.
        doc_stride: When splitting up a long document into chunks, how much stride to take between chunks
        n_best_size: The total number of n-best predictions to generate when looking for an answer
        """
        
    def train(self, train_data_path : str, output_path : str, model_name: Optional[str]=None, valn_path :Optional[str]=None, **kwargs):
        """Function used to fine tune any huggingface qna models.

        Parameters
        ----------
        train_data_path: str
            Training data file/path of csv or json file
        output_path: str
            Output directory to store the finetuned model
        model_name: None, optional (str)
            If set as `None`, default model : "deepset/tinyroberta-squad2" is considered
        valn_path : None, optional (str)
            An optional validation data file/path to evaluate the perplexity on (a csv or json file)
        kwargs: default parameters
            Any default parameters can be used for fine tuning the model. eg:learning_rate,doc_stride etc..

        Returns
        -------
            None

        Examples
        --------
        >>> from question_answer import QnA
        >>> qna=QnA()
        >>> finetuned_model=qna.train(train_data_path='train.json/csv',output_path='output_directory',model_name='deepset/roberta-base-squad2',valn_path='validate.json/csv')
        """
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
                print("Trainer output: ",trainer.stdout)
            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))

            # saving predicted output as jsonl for validation file
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
                print("Trainer output: ",trainer.stdout)

            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))


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
        question : str = None,
        context : str = None,
        model_name : Optional[str] = None,
        output_path : str = None,
        test_path: str = None,
        is_train=False, 
        **kwargs):

        """Inference Function to test an input/file.

        Parameters
        ----------
        question: str
            An input string
        context: str
            An input string
        test_path: str
            Test data file/path to evaluate the perplexity on (a csv or json file)
        output_path: str
            Output directory to store the test file predicted results
        model_name: None, optional (str)
            If set as `None`, default model : "deepset/tinyroberta-squad2" is considered
        is_train: False
            If is_train=False(Default) used to save predicted test result in jsonl format or if is_train=True used to save predicted train result in jsonl format.
        kwargs: default parameters
            Any default parameters can be used for prediction. eg:learning_rate,doc_stride etc..

        Returns
        -------
        dict | output directory
            Dict containing question,context,predicted_answer,model_name,score,answer_start,answer_end values.
            Predicted test file are stored in output directory.

        Examples
        --------
        >>> from question_answer import QnA
        >>> qna=QnA()
        >>> qna.predict(test_path="test.json/csv",output_path="output_dir",model_name='deepset/roberta-base-squad2')
        >>> # or
        >>> qna.predict(question=question,context=context,doc_stride=128,max_answer_length=20,learning_rate=3e-5,n_best_size=20)
        """
        if not model_name:
            model_name = "deepset/tinyroberta-squad2"
            print("No model name specified. Considering default model : deepset/tinyroberta-squad2")
      
        print("Executed")
        print('Model Name: {}'.format(model_name))

        # Predicting answers when context and question are string dtype
        if isinstance(context,str) or isinstance(question,str): 
            if not context or not question:
                raise NameError(
                    "Please enter the context/question as a string"
                )  
            #Infer QA model with transformers library using question-answering pipeline      
            question_answerer = pipeline('question-answering', model=model_name)
            res = question_answerer(question=question,context=context,**kwargs)
            df = pd.DataFrame([res])
            df['context'] = context
            df['question'] = question
            df['model_name'] = model_name
            df.rename(columns = {'answer':'predicted_answer'}, inplace = True)
            df= df[['question','context','predicted_answer','model_name','score','start','end']]
            # Create dictionary
            dictionary = df.to_dict(orient="records")
            # return dictionary
            return dictionary

        #Predicting answers when context and question are specified in csv/json file
        else:
            if not test_path or not output_path:
                raise NameError(
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
                "--output_dir",
                self.output_path,
                "--overwrite_output_dir",  # to overwrite the existing files
            ]

            # Start the training
            try:
                evaluator = subprocess.run(
                    model_params,
                    # shell=True
                    check=True,
                    capture_output=True,
                )
                print("Evaluation output: ",evaluator.stdout)

            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))
            
            # saving predicted output as jsonl for test file
            generate_jsonl(
                self.test_path,
                self.output_path,
                self.model_name,
                "train" if is_train else "test",
            )