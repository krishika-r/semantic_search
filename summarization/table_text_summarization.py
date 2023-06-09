# Table Summarizer - ToTTo data
"""The script is for training and inferencing text and table summarizer
"""

import os
import subprocess
from subprocess import CalledProcessError
import pandas as pd
import itertools
import glob
import json
from transformers import pipeline
from typing import Optional
from helper import generate_jsonl,create_table

class Summarizer:
    """
    Summarization is the task of producing a shorter version of a document while preserving its important information. 
    Some models can extract text from the original input, while other models can generate entirely new text

    Examples
    --------
    >>> from table_text_summarization import Summarizer
    >>> model = Summarizer(summary_type="table")
    >>> model = Summarizer(summary_type="text") or model = Summarizer()
    """
    def __init__(self,summary_type: str = "text"):
        self.summary_type = summary_type.lower()
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.num_train_epochs = 2
        self.max_source_length = 1024
        self.max_target_length = 128
        self.learning_rate = 3e-5 
        self.t5_params = [
            "--source_prefix",
            "summarize: ",
        ]

        """Summarizer initialization

        Parameters
        ----------
        summary_type : Specify table/text. Default = "text" 
        per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size: The batch size per GPU/TPU core/CPU for evaluation
        num_train_epochs: Total number of training epochs to perform.
        max_source_length: The maximum total input sequence length after tokenization. Sequences longer 
                           than this will be truncated, sequences shorter will be padded.
        max_target_length: The maximum total sequence length for target text after tokenization. Sequences longer
                           than this will be truncated, sequences shorter will be padded.
        learning_rate: The initial learning rate for ADAM
        t5_params: 
            source_prefix: A summarize prefix to be added before every source text for T5 models.
        """

    def pre_process(self, data: str = None, data_type: str = None):
        """Function used to preprocess the train/validation/test data using lattice.

        Parameters
        ----------
        data: str
            Train/validation/test data folder
        data_type: str
            Specifiy string input as train/validation/test

        Returns
        -------
            None

        Examples
        --------
        >>> from table_text_summarization import Summarizer
        >>> model = Summarizer(summary_type="table")
        >>> model.pre_process(data='train_folder',data_type='train')
        """
        if self.summary_type=="table":
            random_num = 111234
            json_file = []
            file_list = glob.glob(data + "/*.csv")  # ext placeholder
            if file_list:
                for file in file_list:
                    temp_json = {}
                    #reading a csv file
                    df = pd.read_csv(file)
                    df.columns = [column.replace("_", " ") for column in df.columns]
                    #converting input df to jsonl format
                    parent_list = create_table(df)

                    summary_file_name = os.path.splitext(file)[0]
                    if os.path.exists(f'{summary_file_name}.jsonl'):
                        #reading jsonl file
                        summary_df = pd.read_json(f'{summary_file_name}.jsonl',lines=True)
                        if 'summary' in summary_df.columns:
                            actual_summary = summary_df.summary.values[0]
                            print(actual_summary)
                        else:
                            actual_summary = ' '
                        if 'highlighted_cells' in summary_df.columns:
                            highlighted_cells = summary_df.highlighted_cells.values[0]
                            print(highlighted_cells)
                        else:
                            highlighted_cell = list(
                                    itertools.product(range(1, df.shape[0] + 1), range(df.shape[1]))
                                )
                            highlighted_cells = [list(i) for i in highlighted_cell]
                    else:
                        actual_summary = ' '
                        highlighted_cell = list(
                                itertools.product(range(1, df.shape[0] + 1), range(df.shape[1]))
                            )
                        highlighted_cells = [list(i) for i in highlighted_cell]

                    temp_json["table"] = parent_list
                    temp_json["table_webpage_url"] = ""
                    temp_json["table_page_title"] = ""
                    temp_json["table_section_title"] = ""
                    temp_json["table_section_text"] = ""
                    temp_json["highlighted_cells"] = highlighted_cells
                    temp_json["example_id"] = int(random_num)
                    temp_json["sentence_annotations"] = [
                                {
                                    "original_sentence": "",
                                    "sentence_after_deletion": "",
                                    "sentence_after_ambiguity": "",
                                    "final_sentence": actual_summary,
                                }
                            ]

                    random_num += 1
                    json_file.append(temp_json)

                processed_data=os.path.join(data,'processed')
    
                # creating a folder to save the processed data 
                os.makedirs(processed_data, exist_ok=True)
                with open(os.path.join(processed_data, f"processed_data_{data_type}.jsonl"), "w") as outfile:
                    for entry in json_file:
                        json.dump(entry, outfile)
                        outfile.write("\n")

                #Lattice preprocessing
                process_params = [
                    "python",
                    "Lattice/preprocess_data.py",
                    "--input_path",
                    os.path.join(processed_data, f"processed_data_{data_type}.jsonl"),
                    "--output_path",
                    os.path.join(processed_data, f"{data_type}_data_linearized.jsonl"),
                ]
    
                subprocess.run(
                    process_params,
                    check=True,
                    capture_output=True,
                )
                print("Executed Lattice...")
    
                lattice_output = pd.read_json(
                    os.path.join(processed_data, f"{data_type}_data_linearized.jsonl"), lines=True
                )  # jsonl is the output from lattice step 1
    
                #saving the Lattice output in a dataframe
                model_input = pd.DataFrame()
                if "sentence_annotations" in lattice_output.columns:
                    model_input["text"] = lattice_output["subtable_metadata_str"]
                    model_input["summary"] = (
                        lattice_output["sentence_annotations"]
                        .apply(lambda x: x[0]["final_sentence"])
                        .values
                    )
                else:
                    model_input["text"] = lattice_output["subtable_metadata_str"]
                    model_input["summary"] = " "
    
                #saving the final preprocessed data to json format
                model_input.to_json(
                    os.path.join(processed_data, f"{data_type}_data.json"),
                    orient="records",
                )  # you can pass this {data_type}_data.json to the model
                print("Pre-processing done...")
            else:
                print("Given folder is empty.Please place input csv files...exiting....")

    def train(self,train_data_path : str, output_path : str, model_name: Optional[str]=None, valn_path :Optional[str]=None, model_type :str =None, **kwargs):

        """Function used to fine tune huggingface summarization models.

        Parameters
        ----------
        train_data_path: str
            Training data file/path of csv or json file
        output_path: str
            Output directory to store the finetuned model
        model_name: None, optional (str)
            If set as `None`, default model : "facebook/bart-large-cnn" is considered
        valn_path : None, optional (str)
            An optional validation data file/path to evaluate the perplexity on (a csv or json file)
        model_type : None
            If set as `None`, t5 params are not taken into consideration
        kwargs: default parameters
            Any default parameters can be used for fine tuning the model. eg:learning_rate,max_source_length etc..

        Returns
        -------
            None

        Examples
        --------
        >>> from table_text_summarization import Summarizer
        >>> model = Summarizer(summary_type="table")
        >>> finetuned_model=model.train(train_data_path='train.json/csv',output_path='output_directory',model_name='t5-small',model_type='t5',valn_path='validate.json/csv')
        """

        self.train_data_path = train_data_path
        self.valn_path = valn_path
        self.output_path = output_path
        self.model_name = model_name
        self.model_type = model_type

        if not model_name:
            self.model_name = "facebook/bart-large-cnn"
            print("No model name specified. Considering default model : facebook/bart-large-cnn")

        if not train_data_path or not output_path:
            raise NameError("Provide train_data_path/output_path in the input")
       
        if 'per_device_train_batch_size' in kwargs:
            self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
        if 'per_device_eval_batch_size' in kwargs:
            self.per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size')
        if 'max_source_length' in kwargs:
            self.max_source_length = kwargs.get('max_source_length')
        if 'max_target_length' in kwargs:
            self.max_target_length = kwargs.get('max_target_length')
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
        if 'num_train_epochs' in kwargs:
            self.num_train_epochs = kwargs.get('num_train_epochs')
        
        print("Executed")
        print('Model Name: {}'.format(self.model_name))

        model_params = [
            "python",
            "run_summarization.py",
            "--model_name_or_path",
            self.model_name,
            "--train_file",
            self.train_data_path,
            "--do_train",
            "--text_column",
            "text",
            "--summary_column",
            "summary",
            f"--per_device_train_batch_size={self.per_device_train_batch_size}",
            f"--per_device_eval_batch_size={self.per_device_eval_batch_size}",
            f"--max_source_length={self.max_source_length}",
            f"--max_target_length={self.max_target_length}",
            f"--learning_rate={self.learning_rate}",
            f"--num_train_epochs={self.num_train_epochs}",
            "--predict_with_generate",
        ]

        if model_type.lower() == "t5": #if model type is t5
            model_params += self.t5_params
            

        # command line argument for validation
        validation_cli = [
            "--do_eval",
            "--validation_file",
            self.valn_path,
            "--output_dir",
            os.path.join(self.output_path,"training"),
            "--overwrite_output_dir", # to overwrite the existing files
        ]

        # command line argument for train
        train_cli = [
            "--output_dir",
            os.path.join(self.output_path,"training"),
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

        # saving the finetuned model
        trained_model = os.path.join(self.output_path,"training")

        if self.train_data_path:
            print("train prediction")
            #train prediction
            self.predict(
                model_name=trained_model,
                output_path=os.path.join(self.output_path,"prediction"),
                test_path=self.train_data_path,
                datatype='train',
                model_type=self.model_type,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
            )

        if self.valn_path:
            print("validation prediction")
            #validation prediction
            self.predict(
                model_name=trained_model,
                output_path=os.path.join(self.output_path,"validation"),
                test_path=self.valn_path,
                datatype='validation',
                model_type=self.model_type,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
            )
        
        return
            

    def predict(
        self,
        context: str = None,
        output_path : str =None,
        model_name : Optional[str]=None,
        test_path: str = None,
        datatype: str =None,
        model_type: str = None,
        **kwargs
    ):
        """Inference Function to test an input/file.

        Parameters
        ----------
        context: str
            An input string
        test_path: str
            Test data file/path to evaluate the perplexity on (a csv or json file)
        output_path: str
            Output directory to store the test file predicted results
        model_name: None, optional (str)
            If set as `None`, default model : "facebook/bart-large-cnn" is considered
        datatype: str 
            If datatype='test' used to save predicted test result in jsonl format or if datatype='train' used to save predicted train result in jsonl format or if datatype='validation' used to save predicted validation result in jsonl format.
        model_type : None
            If set as `None`, t5 params are not taken into consideration
        kwargs: default parameters
            Any default parameters can be used for prediction. eg:learning_rate,doc_stride etc..

        Returns
        -------
        dict | output directory
            Dict containing context,predicted_summary,model_name
            Predicted test file are stored in output directory.

        Examples
        --------
        >>> from table_text_summarization import Summarizer
        >>> model = Summarizer() # default it takes text as summary type
        >>> model.predict(test_path="test.json/csv",output_path="output_dir",model_name='t5-small',model_type='t5')
        >>> # or
        >>> model.predict(context=context,model_name='t5-small',min_length=5, max_length=20)

        """
        #getting the model path
        if not model_name:
            model_name = "facebook/bart-large-cnn"
            print("No model name specified. Considering default model : facebook/bart-large-cnn")
        
        #Predicting answers when context is provided as a text and summary_type is text
        if self.summary_type=="text":
            if isinstance(context,str):
                #Infer Summarization model with transformers library using summarization pipeline
                summarizer = pipeline("summarization", model=model_name,**kwargs)
                generated_sum = summarizer(str(context))[0]["summary_text"]
                df = pd.DataFrame([generated_sum],columns=['predicted_summary'])
                df['context'] = context
                df['model_name'] = model_name
                df= df[['context','predicted_summary','model_name']]
                # Create dictionary
                dictionary = df.to_dict(orient="records")
                # Return dictionary
                return dictionary

        if self.summary_type=="text" or self.summary_type=="table":
            #Predicting answers when context is specified in csv/json file
            if not test_path or not output_path:
                raise NameError(
                    "Please enter the test path/output path"
                )
            self.test_path = test_path
            self.output_path = output_path
            self.model_name = model_name
            self.model_type = model_type

            if 'per_device_train_batch_size' in kwargs:
                self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
            if 'per_device_eval_batch_size' in kwargs:
                self.per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size')
            if 'max_source_length' in kwargs:
                self.max_source_length = kwargs.get('max_source_length')
            if 'max_target_length' in kwargs:
                self.max_target_length = kwargs.get('max_target_length')
            if 'learning_rate' in kwargs:
                self.learning_rate = kwargs.get('learning_rate')
            if 'num_train_epochs' in kwargs:
                self.num_train_epochs = kwargs.get('num_train_epochs')

            model_params = [
                "python",
                "run_summarization.py",
                "--model_name_or_path",
                self.model_name,
                "--text_column",
                "text",
                "--do_predict",
                "--test_file",
                self.test_path,
                "--predict_with_generate",
                "--overwrite_output_dir",  # to overwrite the existing files
                "--output_dir",
                self.output_path,
                f"--per_device_train_batch_size={self.per_device_train_batch_size}",
                f"--per_device_eval_batch_size={self.per_device_eval_batch_size}",
                f"--max_source_length={self.max_source_length}",
                f"--max_target_length={self.max_target_length}",
                f"--learning_rate={self.learning_rate}",
                f"--num_train_epochs={self.num_train_epochs}",
            ]

            if model_type.lower() == "t5": # model_type is t5
                model_params += self.t5_params

            # Start the training
            try:       
                evaluator = subprocess.run(
                    model_params,
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
                self.summary_type,
                datatype if datatype else "test",
            )