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

    def pre_process(self, data_path: str = None, data_type: str = None):
        """Function used to preprocess the train/validation/test data using lattice.

        Parameters
        ----------
        data_path: str
            Train/validation/test data folder
        data_type: str
            Specifiy string input as train/validation/test

        Returns
        -------
            None

        Examples
        --------
        >>> from summarization_trainer import Summarizer
        >>> model = Summarizer(summary_type="table")
        >>> model.pre_process(data_path='train_folder',data_type='train')
        """
        if self.summary_type=="table":
            random_num = 111234
            json_file = []
            file_list = glob.glob(data_path + "/*.csv")  # ext placeholder
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

                processed_data=os.path.join(data_path,'processed')
    
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
        else:
            print("No preprocessing step for text summarization")
            return 

    def train(
        self,
        train_data_path : str = None,
        output_path : str = None,
        model_type :str = None,
        train_prediction :str = False,
        val_prediction :Optional[str] = False, 
        model_name: Optional[str] = None, 
        valn_path :Optional[str] = None, 
        **kwargs
    ):

        """Function used to fine tune huggingface summarization models.

        Parameters
        ----------
        train_data_path: None (str)
            Training data file/path of csv or json file
        output_path: None (str)
            Output directory to store the finetuned model
        model_name: None, optional (str)
            If set as 'None', default model : "facebook/bart-large-cnn" is considered
        model_type : None (str)
            If set as 't5', t5 params is taken into consideration. If set as others, t5 params are not considered
        train_prediction : False (str)
            If set as 'True', prediction for train data is calculated.
        val_prediction : False, optional (str)
            If set as 'True', prediction for validation data is calculated.
        valn_path : None, optional (str)
            An optional validation data file/path to evaluate the perplexity on (a csv or json file)
        kwargs: default parameters
            Any default parameters can be used for fine tuning the model. eg:learning_rate,max_source_length etc..

        Returns
        -------
            None

        Examples
        --------
        >>> from table_text_summarization import Summarizer
        >>> model = Summarizer(summary_type="table")
        >>> finetuned_model=model.train(train_data_path='train.json/csv',output_path='output_directory',model_name='t5-small',model_type='t5',valn_path='validate.json/csv',train_prediction=True,val_prediction=True)
        """

        self.train_data_path = train_data_path
        self.valn_path = valn_path
        self.output_path = output_path
        self.model_name = model_name
        self.model_type = model_type
        self.train_prediction = train_prediction
        self.val_prediction = val_prediction

        if not model_name:
            self.model_name = "facebook/bart-large-cnn"
            self.model_type = "others"
            print("No model name specified. Considering default model : facebook/bart-large-cnn")

        if not train_data_path or not output_path:
            raise NameError("Provide train_data_path and output_path in the input")
       
        if not model_type:
            raise NameError("Provide model_type.If t5 model then model_type='t5' else 'others'.\nPossible values:\n1.'t5' \n 2.'others'")

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

        # Defining user specified kwargs to model params 
        for key, value in kwargs.items():
            if key not in ['per_device_train_batch_size','per_device_eval_batch_size','max_source_length',
            'max_target_length','learning_rate','num_train_epochs']:
                model_params.append(f"--{key}={value}")

        if "t5" in model_type.lower(): #if model type is t5
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

        # extracting user defined kwargs alone by removing these 6 default kwargs 
        keys_to_delete = ['per_device_train_batch_size','per_device_eval_batch_size','max_source_length',
            'max_target_length','learning_rate','num_train_epochs']
        for key in keys_to_delete:
            if key in kwargs:
                del kwargs[key]

        if self.train_prediction==True:
            print("train prediction")
            #prediction folder path
            prediction_path = os.path.join(self.output_path,"prediction")
            #train prediction
            self.predict(
                model_name=trained_model,
                output_path=os.path.join(prediction_path,"train"),
                test_path=self.train_data_path,
                datatype='train',
                model_type=self.model_type,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                **kwargs
            )

        if self.val_prediction==True:
            if not valn_path:
                raise NameError("Provide validation data path in the input for validation prediction")
            print("validation prediction")
            #prediction folder path
            prediction_path = os.path.join(self.output_path,"prediction")
            #validation prediction
            self.predict(
                model_name=trained_model,
                output_path=os.path.join(prediction_path,"validation"),
                test_path=self.valn_path,
                datatype='validation',
                model_type=self.model_type,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                **kwargs
            )
        return
            
    def predict(
        self,
        model_type: str = None,
        context: str = None,
        output_path : str =None,
        model_name : Optional[str]=None,
        test_path: str = None,
        datatype: str =None,
        **kwargs
    ):
        """Inference Function to test an input/file.

        Parameters
        ----------
        model_type : None (str)
            If set as `t5`, t5 params is taken into consideration. If set as others, t5 params are not considered
        context: None (str)
            An input string
        test_path: None (str)
            Test data file/path to evaluate the perplexity on (a csv or json file)
        output_path: None (str)
            Output directory to store the test file predicted results
        model_name: None, optional (str)
            If set as `None`, default model : "facebook/bart-large-cnn" is considered
        datatype: None (str)
            If datatype='test' used to save predicted test result in jsonl format or if datatype='train' used to save predicted train result in jsonl format or if datatype='validation' used to save predicted validation result in jsonl format.
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
        >>> model.predict(context=context,model_name='t5-small',min_length=5, max_length=20,model_type='t5')

        """
        #getting the model path
        if not model_name:
            self.model_name = "facebook/bart-large-cnn"
            self.model_type = "others"
            print("No model name specified. Considering default model : facebook/bart-large-cnn")

        if not model_type:
            raise NameError("Provide model_type.If t5 model then model_type='t5' else 'others'.\nPossible values:\n1.'t5' \n 2.'others'")

        #Predicting answers when context is provided as a text and summary_type is text
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

        #Predicting answers when context is specified in csv/json file
        if not test_path or not output_path:
            raise NameError(
                "Please enter the test path and output path"
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

        # Defining user specified kwargs to model params 
        for key, value in kwargs.items():
            if key not in ['per_device_train_batch_size','per_device_eval_batch_size','max_source_length',
            'max_target_length','learning_rate','num_train_epochs']:
                model_params.append(f"--{key}={value}")

        if "t5" in model_type.lower(): # model_type is t5
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
        return 