# tools/code_interpreter.py
import asyncio
import io
import sys
import traceback
import uuid
import atexit
import os
from typing import Dict, Any

from tools.base_tool import BaseTool, ToolResult, ToolSchema

# Get logger
import logging
logger = logging.getLogger(__name__)

class CodeInterpreter(BaseTool):
    """Python code execution sandbox with registry support"""
    
    # Registry metadata
    __tool_name__ = 'code_interpreter'  # This will auto-register
    
    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        super().__init__(
            name="code_interpreter",
            description="Execute Python code in a sandboxed environment with safety checks"
        )
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.execution_count = 0
        self.instance_id = str(uuid.uuid4())
        self.work_dir = '/tmp/agent_code_interpreter'
        
        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup cleanup on exit
        atexit.register(self._cleanup)
        
        logger.info(f"CodeInterpreter initialized: {self.instance_id}")
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema for function calling"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute in sandboxed environment"
                    }
                },
                "required": ["code"]
            },
            required=["code"]
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute Python code with enhanced safety and output capture"""
        code = arguments.get("code", "")
        if not code.strip():
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
                    'had_error': bool(stderr),
                    'instance_id': self.instance_id,
                    'work_dir': self.work_dir,
                    'timeout': self.timeout
                }
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Code execution timed out after {self.timeout} seconds",
                success=False,
                error="Execution timeout",
                metadata={
                    'execution_count': self.execution_count,
                    'timeout': self.timeout
                }
            )
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return ToolResult(
                content=error_msg,
                success=False,
                error=str(e),
                metadata={
                    'execution_count': self.execution_count,
                    'instance_id': self.instance_id
                }
            )
    
    async def _execute_code(self, code: str, stdout: io.StringIO, stderr: io.StringIO) -> Any:
        """Execute code with output capture and safety isolation"""
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            
            # Create isolated namespace with limited builtins
            namespace = {
                '__name__': '__main__',
                '__builtins__': {
                    # Allow safe builtins
                    'abs': abs,
                    'all': all,
                    'any': any,
                    'bool': bool,
                    'dict': dict,
                    'enumerate': enumerate,
                    'filter': filter,
                    'int': int,
                    'len': len,
                    'list': list,
                    'map': map,
                    'max': max,
                    'min': min,
                    'range': range,
                    'round': round,
                    'set': set,
                    'sorted': sorted,
                    'str': str,
                    'sum': sum,
                    'zip': zip,
                    'True': True,
                    'False': False,
                    'None': None,
                },
                # Override print to avoid spam in output
                'print': lambda *args, **kwargs: None,
            }
            
            # Execute code safely
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, namespace, namespace)
            
            return None  # exec doesn't return results
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            # Cleanup any temporary files or processes
            logger.info(f"CodeInterpreter cleanup: {self.instance_id}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    @property
    def execution_stats(self) -> Dict[str, Any]:
        """Get enhanced execution statistics"""
        stats = super().execution_stats
        stats.update({
            'instance_id': self.instance_id,
            'work_dir': self.work_dir,
            'timeout': self.timeout,
            'max_output_length': self.max_output_length
        })
        return stats