import pandas as pd
import json
import logging
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flatten_json.log"),
        logging.StreamHandler()
    ]
)

def flatten_json(json_data, parent_key='', sep='_') -> dict:
    """
    Recursively flattens a nested JSON object.

    Args:
        json_data (dict): The JSON object to flatten.
        parent_key (str): The base key to append to.
        sep (str): Separator to use for flattened keys.

    Returns:
        dict: A flattened dictionary.

    Raises:
        TypeError: If json_data is not a dictionary.
    """
    if not isinstance(json_data, dict):
        logging.error("Input data is not a valid dictionary.")
        raise TypeError("Input data must be a dictionary.")

    items = {}
    for k, v in json_data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            logging.debug(f"Processing nested dictionary at key: {new_key}")
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def json_to_csv(json_data: Union[dict, str], output_file: str) -> None:
    """
    Flattens a JSON object, converts it into a Pandas DataFrame, and saves it as a CSV file.

    Args:
        json_data (Union[dict, str]): The JSON object or string to flatten.
        output_file (str): Path to the CSV file to save the flattened data.

    Raises:
        ValueError: If the input JSON string is invalid.
        IOError: If there is an issue saving the file.
    """
    try:
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            logging.info("Parsing JSON string.")
            json_data = json.loads(json_data)

        # Flatten the JSON data
        logging.info("Flattening JSON data.")
        flattened_data = flatten_json(json_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([flattened_data])
        logging.info("Converting flattened data to DataFrame.")

        # Save DataFrame to CSV
        df.to_csv(output_file, index=False)
        logging.info(f"Data successfully saved to {output_file}")

    except json.JSONDecodeError as e:
        logging.error("Invalid JSON string provided.")
        raise ValueError("Invalid JSON string.") from e

    except IOError as e:
        logging.error(f"Error saving file {output_file}.")
        raise IOError(f"Error saving file {output_file}.") from e

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        raise

# Example JSON
json_str = """{
  "Customer": {
    "PrimaryEmailAddr": {
      "Address": "Surf@Intuit.com"
    }, 
    "SyncToken": "0", 
    "domain": "QBO", 
    "GivenName": "Bill", 
    "DisplayName": "Bill's Windsurf Shop", 
    "BillWithParent": false, 
    "FullyQualifiedName": "Bill's Windsurf Shop", 
    "CompanyName": "Bill's Windsurf Shop", 
    "FamilyName": "Lucchini", 
    "sparse": false, 
    "PrimaryPhone": {
      "FreeFormNumber": "(415) 444-6538"
    }, 
    "Active": true, 
    "Job": false, 
    "BalanceWithJobs": 85.0, 
    "BillAddr": {
      "City": "Half Moon Bay", 
      "Line1": "12 Ocean Dr.", 
      "PostalCode": "94213", 
      "Lat": "37.4307072", 
      "Long": "-122.4295234", 
      "CountrySubDivisionCode": "CA", 
      "Id": "3"
    }, 
    "PreferredDeliveryMethod": "Print", 
    "Taxable": false, 
    "PrintOnCheckName": "Bill's Windsurf Shop", 
    "Balance": 85.0, 
    "Id": "2", 
    "MetaData": {
      "CreateTime": "2014-09-11T16:49:28-07:00", 
      "LastUpdatedTime": "2014-09-18T12:56:01-07:00"
    }
  }, 
  "time": "2015-07-23T11:04:15.496-07:00"
}"""

invoice_json = """

{
  "Invoice": {
    "TxnDate": "2014-09-19", 
    "domain": "QBO", 
    "PrintStatus": "NeedToPrint", 
    "SalesTermRef": {
      "value": "3"
    }, 
    "TotalAmt": 362.07, 
    "Line": [
      {
        "Description": "Rock Fountain", 
        "DetailType": "SalesItemLineDetail", 
        "SalesItemLineDetail": {
          "TaxCodeRef": {
            "value": "TAX"
          }, 
          "Qty": 1, 
          "UnitPrice": 275, 
          "ItemRef": {
            "name": "Rock Fountain", 
            "value": "5"
          }
        }, 
        "LineNum": 1, 
        "Amount": 275.0, 
        "Id": "1"
      }, 
      {
        "Description": "Fountain Pump", 
        "DetailType": "SalesItemLineDetail", 
        "SalesItemLineDetail": {
          "TaxCodeRef": {
            "value": "TAX"
          }, 
          "Qty": 1, 
          "UnitPrice": 12.75, 
          "ItemRef": {
            "name": "Pump", 
            "value": "11"
          }
        }, 
        "LineNum": 2, 
        "Amount": 12.75, 
        "Id": "2"
      }, 
      {
        "Description": "Concrete for fountain installation", 
        "DetailType": "SalesItemLineDetail", 
        "SalesItemLineDetail": {
          "TaxCodeRef": {
            "value": "TAX"
          }, 
          "Qty": 5, 
          "UnitPrice": 9.5, 
          "ItemRef": {
            "name": "Concrete", 
            "value": "3"
          }
        }, 
        "LineNum": 3, 
        "Amount": 47.5, 
        "Id": "3"
      }, 
      {
        "DetailType": "SubTotalLineDetail", 
        "Amount": 335.25, 
        "SubTotalLineDetail": {}
      }
    ], 
    "DueDate": "2014-10-19", 
    "ApplyTaxAfterDiscount": false, 
    "DocNumber": "1037", 
    "sparse": false, 
    "CustomerMemo": {
      "value": "Thank you for your business and have a great day!"
    }, 
    "ProjectRef": {
      "value": "39298045"
    }, 
    "Deposit": 0, 
    "Balance": 362.07, 
    "CustomerRef": {
      "name": "Sonnenschein Family Store", 
      "value": "24"
    }, 
    "TxnTaxDetail": {
      "TxnTaxCodeRef": {
        "value": "2"
      }, 
      "TotalTax": 26.82, 
      "TaxLine": [
        {
          "DetailType": "TaxLineDetail", 
          "Amount": 26.82, 
          "TaxLineDetail": {
            "NetAmountTaxable": 335.25, 
            "TaxPercent": 8, 
            "TaxRateRef": {
              "value": "3"
            }, 
            "PercentBased": true
          }
        }
      ]
    }, 
    "SyncToken": "0", 
    "LinkedTxn": [
      {
        "TxnId": "100", 
        "TxnType": "Estimate"
      }
    ], 
    "BillEmail": {
      "Address": "Familiystore@intuit.com"
    }, 
    "ShipAddr": {
      "City": "Middlefield", 
      "Line1": "5647 Cypress Hill Ave.", 
      "PostalCode": "94303", 
      "Lat": "37.4238562", 
      "Long": "-122.1141681", 
      "CountrySubDivisionCode": "CA", 
      "Id": "25"
    }, 
    "EmailStatus": "NotSet", 
    "BillAddr": {
      "Line4": "Middlefield, CA  94303", 
      "Line3": "5647 Cypress Hill Ave.", 
      "Line2": "Sonnenschein Family Store", 
      "Line1": "Russ Sonnenschein", 
      "Long": "-122.1141681", 
      "Lat": "37.4238562", 
      "Id": "95"
    }, 
    "MetaData": {
      "CreateTime": "2014-09-19T13:16:17-07:00", 
      "LastUpdatedTime": "2014-09-19T13:16:17-07:00"
    }, 
    "CustomField": [
      {
        "DefinitionId": "1", 
        "StringValue": "102", 
        "Type": "StringType", 
        "Name": "Crew #"
      }
    ], 
    "Id": "130"
  }, 
  "time": "2015-07-24T10:48:27.082-07:00"
}

"""

# Specify the output CSV file path
output_csv = "flattened_data_inovice_2.csv"

# Flatten JSON and save to CSV
try:
    json_to_csv(invoice_json, output_csv)
except Exception as e:
    logging.error(f"Failed to process JSON: {e}")
