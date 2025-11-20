# tools/code_interpreter.py
import asyncio
import io
import sys
import traceback
from typing import Dict, Any

from tools.base_tool import BaseTool, ToolResult, ToolSchema

class CodeInterpreter(BaseTool):
    """Python code execution sandbox"""
    
    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        super().__init__(
            name="code_interpreter",
            description="Execute Python code in a sandboxed environment"
        )
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.execution_count = 0
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            },
            required=["code"]
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute Python code"""
        code = arguments.get("code", "")
        if not code:
            return ToolResult(
                content="No code provided",
                success=False,
                error="No code provided"
            )
        
        self.execution_count += 1
        
        # Create execution environment
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Execute code with timeout
            result = await asyncio.wait_for(
                self._execute_code(code, stdout_buffer, stderr_buffer),
                timeout=self.timeout
            )
            
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # Combine output
            if result is not None:
                output = f"Result: {result}"
            else:
                output = stdout if stdout else "(No output)"
            
            if stderr:
                output += f"\nErrors:\n{stderr}"
            
            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            return ToolResult(
                content=output,
                success=True,
                metadata={
                    'execution_count': self.execution_count,
                    'output_length': len(output),
                    'had_error': bool(stderr)
                }
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Code execution timed out after {self.timeout} seconds",
                success=False,
                error="Execution timeout"
            )
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return ToolResult(
                content=error_msg,
                success=False,
                error=str(e),
                metadata={'execution_count': self.execution_count}
            )
    
    async def _execute_code(self, code: str, stdout: io.StringIO, stderr: io.StringIO) -> Any:
        """Execute code with output capture"""
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            
            # Create isolated namespace
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                # Add some safe builtins
                'print': lambda *args, **kwargs: None,  # Override print to avoid spam
            }
            
            # Execute code
            result = eval(code, namespace, namespace)
            
            # If it's an expression that returns a value, return it
            # If it's a statement, return None
            return result
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Register the tool
from tools.base_tool import ToolRegistry

@ToolRegistry.register('code_interpreter')
class RegisteredCodeInterpreter(CodeInterpreter):
    pass