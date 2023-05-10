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
    def __init__(self,summary_type= "text"):
        self.summary_type = summary_type.lower()
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.max_source_length = 1024
        self.max_target_length = 128
        self.learning_rate = 3e-5 
        self.t5_params = [
            "--source_prefix",
            "summarize: ",
        ]

    def pre_process(self, data=None,data_type=None):
      random_num = 111234
      json_file = []
      file_list = glob.glob(data + "/*.csv")  # ext placeholder
      if file_list:
        for file in file_list:
          temp_json = {}
          df = pd.read_csv(file)
          df.columns = [column.replace("_", " ") for column in df.columns]

          parent_list = create_table(df)

          summary_file_name = os.path.splitext(file)[0]
          if os.path.exists(f'{summary_file_name}.jsonl'):
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

      model_input.to_json(
          os.path.join(processed_data, f"{data_type}_data.json"),
          orient="records",
      )  # you can pass this {data_type}_data.json to the model
      print("Pre-processing done...")

    def train(self,train_data_path : str, output_path : str, model_name: Optional[str]=None, valn_path :Optional[str]=None, **kwargs):

        self.train_data_path = train_data_path
        self.valn_path = valn_path
        self.output_path = output_path
        self.model_name = model_name

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
            "--predict_with_generate",
        ]

        if self.model_name == "t5-small": #if model name is t5-small
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
            except CalledProcessError as err:
                print("error message",err)
                print(err.stderr.decode('utf8'))

        print("Trainer output: ",trainer.stdout)

        if self.train_data_path:
            print("train prediction")
          #train prediction
            self.predict(
                model_name=self.model_name,
                output_path=os.path.join(self.output_path,"prediction"),
                test_path=self.train_data_path,
                is_train=True,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate
            )

        if self.valn_path:
            print("val prediction")
          #val prediction
            self.predict(
                model_name=self.model_name,
                output_path=os.path.join(self.output_path,"validation"),
                test_path=self.valn_path,
                is_train=True,
                per_device_train_batch_size=self.per_device_train_batch_size, 
                per_device_eval_batch_size= self.per_device_eval_batch_size,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
                learning_rate=self.learning_rate
            )
        
        return
            

    def predict(
        self,
        context: str = None,
        output_path : str =None,
        model_name : Optional[str]=None,
        test_path: str = None,
        is_train=False,
        **kwargs
    ):

        #getting the model path
        if not model_name:
            model_name = "facebook/bart-large-cnn"
            print("No model name specified. Considering default model : facebook/bart-large-cnn")

        if isinstance(context,str):
            summarizer = pipeline("summarization", model=model_name,**kwargs)
            generated_sum = summarizer(str(context))[0]["summary_text"]
            df = pd.DataFrame([generated_sum],columns=['predicted_summary'])
            df['context'] = context
            df['model_name'] = model_name
            df= df[['context','predicted_summary','model_name']]
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
            ]

            if self.model_name == "t5-small":  # if model name is t5 small
                model_params += self.t5_params

            # Start the training
            try:       
                evaluator = subprocess.run(
                    model_params,
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
                self.summary_type,
                "train" if is_train else "test",
            )

            print("Evaluation output: ",evaluator.stdout)

