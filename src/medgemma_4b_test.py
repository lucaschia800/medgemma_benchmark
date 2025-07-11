import transformers
import torch
import json
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from huggingface_hub import login

token = os.getenv("HUGGINGFACE_TOKEN")

login(token)

processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it", device_map="auto")

print("Device map:")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

text = """


You are an expert medical AI assistant specializing in clinical data extraction. Your task is to analyze the provided medical document and extract all relevant clinical information, formatting it as a single JSON object that strictly conforms to the provided Pydantic schemas.

### Core Directives & Rules

1.  **Holistic Analysis**: Analyze the entire document at once to understand the full context before generating the output.
2.  **Evidence is Paramount**: For every resource you create, the resource_evidence field MUST contain the exact, verbatim text from the document that supports the extraction.
3.  **Handle Negations Correctly**: If the document states a patient denies a symptom or is not taking a medication, you MUST create the corresponding resource and set its negation field to true.

---
### Resource Extraction Guidelines

You will extract information for the following four resource types based on their definitions and instructions.

#### 1. Condition Resources

* **Definition**: A Condition represents a diagnosis, problem, or ongoing clinical condition that has risen to a level of concern. It is a medical condition determined as the cause of symptoms and may be found through physical findings, lab/radiology reports, or other means. This resource is for diagnoses, not the subjective or objective information that leads to them.
* **Instructions**:
    * Identify all primary conditions, problems, and diagnoses.
    * Assign a sequential priority_id to each Condition, starting with 1 for the most important condition related to the visit's primary purpose.

#### 2. Observation Resources

* **Definition**: An Observation records measurements, test results, or point-in-time assessments like blood pressure or observed symptoms. It provides the specific subjective and objective data to support a Condition's assertions.
    * **SIGN**: A medical finding reported by the clinician (e.g., "PARASPINAL MUSCLE SPASM: present bilaterally").
    * **SYMPTOM**: A medical issue reported by the patient (e.g., "Patient complains of severe neck, low back, and right shoulder pain").
* **Instructions**:
    * For each sign or symptom, create an Observation resource.
    * You MUST link each Observation to a parent Condition by setting its parent_condition field to the priority_id of the most relevant parent condition.
    * You MUST categorize its relevance by setting the relevance_category field, using the definitions below:
        * **DEFINING**: A hallmark sign/symptom central to the diagnosis (e.g., severe low back pain for Lumbar Radiculopathy).
        * **CORROBORATING**: A common finding that supports the diagnosis but isn't essential (e.g., muscle spasms alongside a disc displacement diagnosis).
        * **CONTEXTUAL**: Clinically relevant information providing general context but not a direct symptom (e.g., Vital Signs like BP or HR).
        * **INCIDENTAL**: A finding mentioned but not clinically relevant to the current problem.

#### 3. AllergyIntolerance Resources

* **Definition**: This represents a clinical assessment of an allergy or intolerance, which is a potential risk for an adverse reaction on future exposure to a substance.
* **Instructions**:
    * Identify ALL stated allergies or intolerances.
    * Populate the allergy_name with the substance that causes the allergy (e.g., "Codeine").

#### 4. MedicationStatement Resources

* **Definition**: This is a record of a medication a patient is, has, or will be consuming.
* **Instructions**:
    * Identify ALL current medications mentioned.
    * If the patient denies taking medication, you MUST create a MedicationStatement resource with negation: true and the medication described as "medication for pain". If no medications are listed, create a resource with negation: true and the medication as "None".

---

Medical Document to Analyze:

        --- PAGE 1 ---

        7/7/25, 2:32 PM
        Print Preview
        HOCKADAY, Stephanie DOB: 12/07/1971 (53 yo F) Acc No. 21815 DOS: 03/05/2025
        REGENACHE™
        HOCKADAY, Stephanie
        53 Y old Female, DOB: 12/07/1971
        Account Number: 21815
        5523 SNAPPING TURTLE RD, BAYTOWN, TX-77523-5059
        Home: 346-549-3100
        Insurance: Self Pay
        Appointment Facility: REGENACHE in the Woodlands
        Mauricio Garcia Jacques, MD
        03/05/2025
        Reason for Appointment

        Consultation
        Assessments

        Muscle spasm - M62.838

        Cervical radiculopathy - M54.12

        Cervical disc displacement - M50.20

        Lumbar radiculopathy - M54.16

        Disc displacement, lumbar - M51.26
        Treatment

        Cervical disc displacement
        Notes: I recommend cervical SCP injection. (1st procedure)
        Clinical Notes:
        Cervical Spine MRI without contrast

        C6/C7 disc herniation (2.5-2.8mm)

        Disc displacement, lumbar
        Notes: I recommend lumbar SCP injection. (2nd procedure)
        Clinical Notes:
        Lumbar Spine MRI without contrast

        L5/S1 disc dehydration and desiccation, disc herniation (3.5mm)
        History of Present Illness
        Constitutional:
        The patient is here today to establish care.
        Patient suffered from MVC on 5/10/2024 and developed symptoms.
        Patient complains of severe neck, low back, and right shoulder pain with 10/10 severity, and mild left shoulder pain with 3/10 severity.
        Low back pain is described as stabbing and aching, while the right shoulder has burning sensation.
        Patient complains of weakness in the whole back and bilateral shoulders, and numbness in the right shoulder.
        Patient reports limitation with daily activities, such as folding clothes. Patient also has difficulties with work and sleep due to pain.
        Patient had 24 sessions of physical therapy which alleviated symptoms. Patient denies taking medication for pain.
        Patient already had injections twice which helped relieve the pain.
        Current Medications
        None
        Progress Note: Mauricio Garcia Jacques, MD 03/05/2025
        Note generated by Clinical Works EMR/PM. Software (www.eClinicalWorks.com)
        1/3

        --- PAGE 2 ---

        7/7/25, 2:32 PM
        Print Preview
        Allergies
        Codeine
        HOCKADAY, Stephanie DOB: 12/07/1971 (53 yo F) Acc No. 21815 DOS: 03/05/2025
        Past Medical History
        Stomach ulcer.
        Bowel and bladder incontinence.
        Review of Systems
        General/Constitutional:
        Denies Change in appetite. Denies Chills. Denies Fatigue. Denies Fever. Denies Headache. Denies Lightheadedness. Sleep disturbance admits. Denies Weight gain.
        Denies Weight loss.
        Musculoskeletal:
        Pain in neck admits. Denies Arthritis. Back problems admits. Denies Carpal tunnel. Denies History of Gout. Joint stiffness admits.
        Denies Cramps. Muscle aches admits. Pain in shoulder(s) affecting the right shoulder. Denies Painful joints. Denies Sciatica. Denies Swollen joints.
        Denies Trauma to arm(s). Denies Trauma to hip(s). Denies Trauma to knee(s). Denies Trauma to ankle(s). Weakness admits.
        Neurologic:
        Tingling/Numbness admits.
        Vital Signs
        HR: 58/min, BP: 157/81 mm Hg, Wt: 170 lbs, BMI: 32.12 Index, Ht: 61 in, Oxygen sat %: 99%.
        Examination
        General Examination:
        GENERAL APPEARANCE: in no acute distress, well developed, well nourished.
        Cervical Spine/Neck:
        RANGE OF MOTION OF NECK: decreased.
        PARASPINAL MUSCLE SPASM: present bilaterally.
        FACET JOINT TENDERNESS present bilaterally.
        Lumbar Spine/Lower back:
        RANGE OF MOTION: decreased.
        PARASPINAL MUSCLE SPASM present bilaterally. FACET LOADING TEST positive bilaterally.
        Visit Codes
        99204 Office Visit, New Pt., Level 4.
        Electronically signed by MAURICIO GARCIA JACQUES, M.D. on 03/13/2025 at 02:35 PM EDT
        Sign off status: Completed
        Progress Note: Mauricio Garcia Jacques, MD 03/05/2025
        Note generated by ClinicalWorks EMR/PM. Software (www.eClinicalWorks.com)
        2/3

        --- PAGE 3 ---

        7/7/25, 2:32 PM
        Print Preview
        HOCKADAY, Stephanie DOB: 12/07/1971 (53 yo F) Acc No. 21815 DOS: 03/05/2025
        REGENACHE in the Woodlands
        200 Valleywood Road
        STE B100
        Spring, TX 77380
        Tel: 832-930-3589
        Fax: 832-241-4756
        Progress Note: Mauricio Garcia Jacques, MD 03/05/2025
        Nole generated by eClinical Works EMR/PM Software (www.eClinicalWorks.com)
        3/3


    JSON Schema:
    {
  "$defs": {
    "AllergyIntolerance": {
      "properties": {
        "resource_type": {
          "const": "AllergyIntolerance",
          "default": "AllergyIntolerance",
          "title": "Resource Type",
          "type": "string"
        },
        "resource_evidence": {
          "description": "The evidence extracted EXACTLY from the document supporting the resource",
          "title": "Resource Evidence",
          "type": "string"
        },
        "negation": {
          "description": "Whether the resource is negated in the document",
          "title": "Negation",
          "type": "boolean"
        },
        "allergy_name": {
          "description": "The name of the allergy or intolerance, extracted as a standalone atomic form",
          "title": "Allergy Name",
          "type": "string"
        },
        "coding": {
          "anyOf": [
            {
              "items": {
                "anyOf": [
                  {
                    "$ref": "#/$defs/SNOMEDCode"
                  },
                  {
                    "$ref": "#/$defs/ICD10Code"
                  },
                  {
                    "$ref": "#/$defs/LOINCCode"
                  },
                  {
                    "$ref": "#/$defs/UnknownCode"
                  }
                ]
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The medical coding of the allergy or intolerance, extracted only if clearly stated.",
          "title": "Coding"
        },
        "onset_age": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The age at which the allergy or intolerance was first noted (e.g., '34 years old').",
          "title": "Onset Age"
        },
        "onset_date": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The date at which the allergy or intolerance was first noted (e.g., '2022-01-15').",
          "title": "Onset Date"
        },
        "last_reaction": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The description or date of the last reaction.",
          "title": "Last Reaction"
        }
      },
      "required": [
        "resource_evidence",
        "negation",
        "allergy_name"
      ],
      "title": "AllergyIntolerance",
      "type": "object"
    },
    "Condition": {
      "properties": {
        "resource_type": {
          "const": "Condition",
          "default": "Condition",
          "title": "Resource Type",
          "type": "string"
        },
        "resource_evidence": {
          "description": "The evidence extracted EXACTLY from the document supporting the resource",
          "title": "Resource Evidence",
          "type": "string"
        },
        "negation": {
          "description": "Whether the resource is negated in the document",
          "title": "Negation",
          "type": "boolean"
        },
        "diagnosis_name": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "description": "The name of the diagnosis, extracted as a standalone atomic form",
          "title": "Diagnosis Name"
        },
        "priority_id": {
          "description": "The priority of this condition in terms of relevancy to the patient visit, 1 being the most relevant identified condition",
          "title": "Priority Id",
          "type": "integer"
        },
        "coding": {
          "anyOf": [
            {
              "items": {
                "anyOf": [
                  {
                    "$ref": "#/$defs/SNOMEDCode"
                  },
                  {
                    "$ref": "#/$defs/ICD10Code"
                  },
                  {
                    "$ref": "#/$defs/LOINCCode"
                  },
                  {
                    "$ref": "#/$defs/UnknownCode"
                  }
                ]
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The medical coding of the condition, extracted only if clearly stated.",
          "title": "Coding"
        }
      },
      "required": [
        "resource_evidence",
        "negation",
        "diagnosis_name",
        "priority_id"
      ],
      "title": "Condition",
      "type": "object"
    },
    "ICD10Code": {
      "properties": {
        "code": {
          "description": "The code, which should only be extracted if clearly stated in the text",
          "title": "Code",
          "type": "string"
        },
        "display": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The display text for the code",
          "title": "Display"
        },
        "system": {
          "const": "http://hl7.org/fhir/sid/icd-10",
          "default": "http://hl7.org/fhir/sid/icd-10",
          "title": "System",
          "type": "string"
        }
      },
      "required": [
        "code"
      ],
      "title": "ICD10Code",
      "type": "object"
    },
    "LOINCCode": {
      "properties": {
        "code": {
          "description": "The code, which should only be extracted if clearly stated in the text",
          "title": "Code",
          "type": "string"
        },
        "display": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The display text for the code",
          "title": "Display"
        },
        "system": {
          "const": "http://loinc.org",
          "default": "http://loinc.org",
          "title": "System",
          "type": "string"
        }
      },
      "required": [
        "code"
      ],
      "title": "LOINCCode",
      "type": "object"
    },
    "MedicationCode": {
      "description": "Code for a medication, accepting RxNorm or SNOMED CT systems.",
      "properties": {
        "code": {
          "description": "The code, which should only be extracted if clearly stated in the text",
          "title": "Code",
          "type": "string"
        },
        "display": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The display text for the code",
          "title": "Display"
        },
        "system": {
          "default": "http://www.nlm.nih.gov/research/umls/rxnorm",
          "enum": [
            "http://www.nlm.nih.gov/research/umls/rxnorm",
            "http://snomed.info/sct"
          ],
          "title": "System",
          "type": "string"
        }
      },
      "required": [
        "code"
      ],
      "title": "MedicationCode",
      "type": "object"
    },
    "MedicationStatement": {
      "properties": {
        "resource_type": {
          "const": "MedicationStatement",
          "default": "MedicationStatement",
          "title": "Resource Type",
          "type": "string"
        },
        "resource_evidence": {
          "description": "The evidence extracted EXACTLY from the document supporting the resource",
          "title": "Resource Evidence",
          "type": "string"
        },
        "negation": {
          "description": "Whether the resource is negated in the document",
          "title": "Negation",
          "type": "boolean"
        },
        "dosage": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The dosage of the medication, extracted as a standalone atomic form",
          "title": "Dosage"
        },
        "medication": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The medication name, extracted as a standalone atomic form",
          "title": "Medication"
        },
        "medication_code": {
          "anyOf": [
            {
              "$ref": "#/$defs/MedicationCode"
            },
            {
              "$ref": "#/$defs/UnknownCode"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The medication code, extracted only if clearly stated.",
          "title": "Medication Code"
        },
        "frequency": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The frequency of the medication, extracted as a standalone atomic form",
          "title": "Frequency"
        },
        "duration": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The duration of the medication, extracted as a standalone atomic form",
          "title": "Duration"
        },
        "strength": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The strength of the medication, extracted as a standalone atomic form",
          "title": "Strength"
        },
        "route": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The route of the medication, extracted as a standalone atomic form",
          "title": "Route"
        }
      },
      "required": [
        "resource_evidence",
        "negation"
      ],
      "title": "MedicationStatement",
      "type": "object"
    },
    "Observation": {
      "properties": {
        "resource_type": {
          "const": "Observation",
          "default": "Observation",
          "title": "Resource Type",
          "type": "string"
        },
        "resource_evidence": {
          "description": "The evidence extracted EXACTLY from the document supporting the resource",
          "title": "Resource Evidence",
          "type": "string"
        },
        "negation": {
          "description": "Whether the resource is negated in the document",
          "title": "Negation",
          "type": "boolean"
        },
        "condition_name": {
          "description": "The name of the sign or symptom, extracted as a standalone atomic form",
          "title": "Condition Name",
          "type": "string"
        },
        "coding": {
          "anyOf": [
            {
              "items": {
                "anyOf": [
                  {
                    "$ref": "#/$defs/SNOMEDCode"
                  },
                  {
                    "$ref": "#/$defs/ICD10Code"
                  },
                  {
                    "$ref": "#/$defs/LOINCCode"
                  },
                  {
                    "$ref": "#/$defs/UnknownCode"
                  }
                ]
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The medical coding of the observation, extracted only if clearly stated.",
          "title": "Coding"
        },
        "parent_condition": {
          "description": "The priority_id of the parent condition of this observation",
          "title": "Parent Condition",
          "type": "integer"
        },
        "relevance_category": {
          "$ref": "#/$defs/RelevanceCategory",
          "description": "Categorize how this observation relates to the parent condition\nDEFINING: A pathognomonic or hallmark sign/symptom that is central to the diagnosis. The diagnosis is unlikely without it.\nCORROBORATING: A common or expected finding that supports the diagnosis but is not essential for it.\nCONTEXTUAL: Clinically relevant information about the patient's general status that is not a direct symptom of the primary diagnosis.\nINCIDENTAL: A finding that is mentioned but is not clinically relevant to the current problem or overall health assessment.\n"
        }
      },
      "required": [
        "resource_evidence",
        "negation",
        "condition_name",
        "parent_condition",
        "relevance_category"
      ],
      "title": "Observation",
      "type": "object"
    },
    "RelevanceCategory": {
      "description": "The category of the relevance of the observation to the parent condition.",
      "enum": [
        "defining",
        "corroborating",
        "contextual",
        "incidental"
      ],
      "title": "RelevanceCategory",
      "type": "string"
    },
    "SNOMEDCode": {
      "properties": {
        "code": {
          "description": "The code, which should only be extracted if clearly stated in the text",
          "title": "Code",
          "type": "string"
        },
        "display": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The display text for the code",
          "title": "Display"
        },
        "system": {
          "const": "http://snomed.info/sct",
          "default": "http://snomed.info/sct",
          "title": "System",
          "type": "string"
        }
      },
      "required": [
        "code"
      ],
      "title": "SNOMEDCode",
      "type": "object"
    },
    "UnknownCode": {
      "properties": {
        "code": {
          "description": "The code, which should only be extracted if clearly stated in the text",
          "title": "Code",
          "type": "string"
        },
        "display": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "The display text for the code",
          "title": "Display"
        },
        "system": {
          "const": "unknown",
          "default": "unknown",
          "title": "System",
          "type": "string"
        }
      },
      "required": [
        "code"
      ],
      "title": "UnknownCode",
      "type": "object"
    }
  },
  "description": "A container/form for the detection of resources in a document",
  "properties": {
    "reasoning": {
      "description": "A brief, high-level summary of the extraction plan and key findings before generating the resources.",
      "title": "Reasoning",
      "type": "string"
    },
    "condition_detection_form": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/Condition"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "description": "A list of conditions detected in the document",
      "title": "Condition Detection Form"
    },
    "medication_detection_form": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/MedicationStatement"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "description": "A list of medications detected in the document",
      "title": "Medication Detection Form"
    },
    "allergy_detection_form": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/AllergyIntolerance"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "description": "A list of allergies detected in the document",
      "title": "Allergy Detection Form"
    },
    "observation_detection_form": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/Observation"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "description": "A list of observations detected in the document",
      "title": "Observation Detection Form"
    }
  },
  "required": [
    "reasoning",
    "condition_detection_form",
    "medication_detection_form",
    "allergy_detection_form",
    "observation_detection_form"
  ],
  "title": "DetectionForm",
  "type": "object"
}
"""

conversation = [
    {
        'role': 'user',
        'content' : [
            {'type' : 'text', 'text' : text}
        ]
        
    }
]
input_device = model.device

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize = True,
    return_dict = True,
    return_tensors = 'pt'
    
)

inputs = {k: v.to(input_device) for k, v in inputs.items()}

input_length = inputs['input_ids'].shape[1]


with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens = 25000)
    outputs = outputs[0][input_length:]

response = processor.decode(outputs, clean_up_tokenization_spaces=True)

print("Raw response:")
print(response)

# Try to parse the response as JSON
try:
    parsed_response = json.loads(response)
    print("\nSuccessfully parsed JSON!")
    
    # Save the parsed JSON with proper formatting
    with open('response.json', 'w') as f:
        json.dump(parsed_response, f, indent=2)
    
    print("Saved formatted JSON to response.json")
    
except json.JSONDecodeError as e:
    print(f"\nFailed to parse response as JSON: {e}")
    print("Saving raw response as text...")
    
    # Save raw response as text if JSON parsing fails
    with open('response.txt', 'w') as f:
        f.write(response)
    
    # Also save the raw response wrapped in JSON for debugging
    with open('response.json', 'w') as f:
        json.dump({"raw_response": response, "error": str(e)}, f, indent=2)







