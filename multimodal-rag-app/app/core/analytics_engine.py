import pandas as pd
import logging
import os
import json
import re
import traceback
import io
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

logger = logging.getLogger(__name__)

class CodeSandbox:
    """Restricted execution environment for dynamic Python logic."""
    # Using \b (word boundaries) prevents blocking words like 'importance'
    FORBIDDEN_PATTERNS = [
        r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\bimport\b", 
        r"\bopen\(", r"\beval\(", r"\bexec\("
    ]

    @staticmethod
    def validate_code(code: str) -> bool:
        """Lints code for dangerous system calls after stripping comments."""
        # Strip comments to prevent false positives from descriptive text
        clean_code = re.sub(r'#.*', '', code)
        
        for pattern in CodeSandbox.FORBIDDEN_PATTERNS:
            if re.search(pattern, clean_code):
                return False
        return True

    @staticmethod
    def execute_safely(df: pd.DataFrame, code: str):
        """Executes validated code in a restricted namespace."""
        if not CodeSandbox.validate_code(code):
            return "Security Violation: Unauthorized system call detected."
        
        local_vars = {"df": df, "pd": pd, "result": None}
        try:
            # Capture stdout for info()
            if "info()" in code:
                buffer = io.StringIO()
                df.info(buf=buffer)
                return buffer.getvalue()

            # Execute in a restricted global scope (no __builtins__)
            exec(code, {"__builtins__": {}}, local_vars)
            return local_vars.get("result", "Calculation complete (no result variable assigned).")
        except Exception:
            return f"Execution Error: {traceback.format_exc().splitlines()[-1]}"

class CSVAnalytics:
    def __init__(self, llm):
        self.llm = llm

    def _get_data_context(self, df: pd.DataFrame):
        """Dynamic schema inspection: No hardcoded column names."""
        context = {
            col: {
                "type": str(df[col].dtype),
                "samples": df[col].dropna().unique()[:3].tolist()
            } for col in df.columns
        }
        return json.dumps(context)

    @traceable(name="Analytics Orchestration")
    async def run_query(self, file_path: str, user_query: str):
        try:
            if not os.path.exists(file_path): 
                return "Data source not found."
            
            df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path)
            df.columns = [c.lower().strip() for c in df.columns]
            
            data_dict = self._get_data_context(df)
            last_error = ""

            for attempt in range(2):
                code_prompt = f"""
                Dataset Schema: {data_dict}
                Available Variable: 'df' (Already loaded in memory)

                Task: Provide ONLY the Python code to answer: "{user_query}"

                STRICT RULES:
                1. Use the existing variable 'df'. DO NOT re-create or redefine 'df'.
                2. DO NOT include any 'import' statements.
                3. DO NOT include comments or explanations.
                4. Assign the final answer to 'result'.
                
                Example of valid output:
                result = df['cost_per_mw'].mean()
                """
                
                res = await self.llm.ainvoke([
                    SystemMessage(content="You are a senior data analyst. Return ONLY executable Python code."), 
                    HumanMessage(content=code_prompt)
                ])
                # Clean markdown blocks if present
                code = re.sub(r"```python|```", "", res.content).strip()
                
                raw_result = CodeSandbox.execute_safely(df, code)
                
                # Check if execution was successful
                if "Violation" not in str(raw_result) and "Error" not in str(raw_result):
                    synthesis_prompt = f"""
                    User Question: {user_query}
                    Data Result: {raw_result}

                    Task: Provide a natural, friendly, and conversational answer.
                    Do not mention 'dataframes' or 'code'. Answer directly.
                    """
                    
                    natural_res = await self.llm.ainvoke([
                        SystemMessage(content="You are a professional assistant."),
                        HumanMessage(content=synthesis_prompt)
                    ])
                    
                    return natural_res.content
                
                last_error = str(raw_result)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            return f"Analysis failed. Last error: {last_error}"

        except Exception as e:
            logger.error(f"Agent Engine Failure: {e}")
            return "Internal system error during analysis."