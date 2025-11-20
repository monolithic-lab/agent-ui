# mcp_hf_server.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from huggingface_hub import HfApi, ModelInfo, DatasetInfo, SpaceInfo
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HuggingFaceHubMCPServer:
    """Unified MCP Server for HuggingFace Hub intelligence"""

    def __init__(self):
        self.app = Server("huggingface-hub")
        self.hf_api = HfApi()
        self._setup_tools()
        logger.info("HuggingFace Hub MCP Server initialized")

    def _setup_tools(self):
        """Setup MCP tools for HuggingFace Hub operations"""

        @self.app.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available HuggingFace Hub tools"""
            return [
                Tool(
                    name="search_models",
                    description="Search for models on HuggingFace Hub with filters like task, library, dataset, etc.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or model name"
                            },
                            "task": {
                                "type": "string", 
                                "description": "Filter by task (e.g., text-classification, text-generation)"
                            },
                            "library": {
                                "type": "string",
                                "description": "Filter by library (e.g., transformers, diffusers)"
                            },
                            "dataset": {
                                "type": "string", 
                                "description": "Filter by trained dataset"
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of results (default: 10)"
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="search_datasets",
                    description="Search for datasets on HuggingFace Hub",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or dataset name"
                            },
                            "task": {
                                "type": "string",
                                "description": "Filter by task category"
                            },
                            "language": {
                                "type": "string",
                                "description": "Filter by language"
                            },
                            "limit": {
                                "type": "number", 
                                "description": "Maximum number of results (default: 10)"
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="search_spaces",
                    description="Search for Spaces on HuggingFace Hub",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or space name"
                            },
                            "sdk": {
                                "type": "string",
                                "description": "Filter by SDK (gradio, streamlit, docker)"
                            },
                            "models": {
                                "type": "string",
                                "description": "Filter by used models"
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of results (default: 10)"
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="get_model_details",
                    description="Get detailed information about a specific model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string", 
                                "description": "Full model ID (e.g., google/gemma-7b)"
                            },
                            "include_files": {
                                "type": "boolean",
                                "description": "Whether to include file list and metadata (default: false)"
                            }
                        },
                        "required": ["model_id"]
                    },
                ),
                Tool(
                    name="get_dataset_details", 
                    description="Get detailed information about a specific dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dataset_id": {
                                "type": "string",
                                "description": "Full dataset ID (e.g., openai/gsm8k)" 
                            },
                            "include_files": {
                                "type": "boolean",
                                "description": "Whether to include file list and metadata (default: false)"
                            }
                        },
                        "required": ["dataset_id"]
                    },
                ),
                Tool(
                    name="get_space_details",
                    description="Get detailed information about a specific Space",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "space_id": {
                                "type": "string",
                                "description": "Full space ID (e.g., huggingface/chat-ui)"
                            },
                            "include_files": {
                                "type": "boolean", 
                                "description": "Whether to include file list and metadata (default: false)"
                            }
                        },
                        "required": ["space_id"]
                    },
                ),
                Tool(
                    name="analyze_repo_structure",
                    description="Analyze the file structure and contents of a repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_id": {
                                "type": "string",
                                "description": "Full repository ID"
                            },
                            "repo_type": {
                                "type": "string", 
                                "description": "Type: model, dataset, or space",
                                "enum": ["model", "dataset", "space"]
                            },
                            "path": {
                                "type": "string",
                                "description": "Specific path to analyze (optional)"
                            }
                        },
                        "required": ["repo_id", "repo_type"]
                    },
                ),
                Tool(
                    name="compare_models",
                    description="Compare multiple models based on metrics and features",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_ids": {
                                "type": "array",
                                "description": "List of model IDs to compare",
                                "items": {"type": "string"}
                            },
                            "criteria": {
                                "type": "array", 
                                "description": "Comparison criteria (downloads, likes, size, etc.)",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["model_ids"]
                    },
                )
            ]

        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: Dict) -> List[TextContent]:
            """Handle tool execution"""
            logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
            try:
                if name == "search_models":
                    result = await self._search_models(**arguments)
                elif name == "search_datasets":
                    result = await self._search_datasets(**arguments)
                elif name == "search_spaces":
                    result = await self._search_spaces(**arguments)
                elif name == "get_model_details":
                    result = await self._get_model_details(**arguments)
                elif name == "get_dataset_details":
                    result = await self._get_dataset_details(**arguments)
                elif name == "get_space_details":
                    result = await self._get_space_details(**arguments)
                elif name == "analyze_repo_structure":
                    result = await self._analyze_repo_structure(**arguments)
                elif name == "compare_models":
                    result = await self._compare_models(**arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
        
                # Format result as JSON
                if isinstance(result, (dict, list)):
                    output_text = json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    output_text = str(result)
        
                logger.info(f"Tool {name} executed successfully")
                return [TextContent(type="text", text=output_text)]
        
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}", exc_info=True)
                error_result = {
                    "error": str(e),
                    "tool": name,
                    "message": f"Failed to execute {name}"
                }
                return [TextContent(
                    type="text",
                    text=json.dumps(error_result, indent=2)
                )]

    async def _search_models(self, query: str, **filters) -> Dict[str, Any]:
        """Search for models with filters"""
        try:
            # Extract limit and set default
            limit = filters.pop('limit', 10)
            
            # Map 'task' to 'pipeline_tag' for HuggingFace API compatibility
            if 'task' in filters:
                filters['pipeline_tag'] = filters.pop('task')
            
            # Remove None values from filters
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            
            models = list(self.hf_api.list_models(
                search=query,
                **clean_filters
            ))
            
            return {
                "count": len(models),
                "results": [
                    {
                        "id": model.id,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "pipeline_tag": model.pipeline_tag,
                        "library_name": model.library_name,
                        "tags": model.tags[:10] if model.tags else [],  # Limit tags for brevity
                        "private": model.private,
                        "created_at": model.created_at.isoformat() if model.created_at else None
                    }
                    for model in models[:int(limit)]
                ]
            }
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return {"error": f"Failed to search models: {str(e)}"}

    async def _search_datasets(self, query: str, **filters) -> Dict[str, Any]:
        """Search for datasets with filters"""
        try:
            # Extract limit and set default
            limit = filters.pop('limit', 10)
            
            # Remove None values from filters
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            
            datasets = list(self.hf_api.list_datasets(
                search=query,
                **clean_filters
            ))
            
            return {
                "count": len(datasets),
                "results": [
                    {
                        "id": dataset.id,
                        "author": dataset.author,
                        "downloads": dataset.downloads,
                        "likes": dataset.likes,
                        "tags": dataset.tags[:10] if dataset.tags else [],  # Limit tags for brevity
                        "private": dataset.private,
                        "created_at": dataset.created_at.isoformat() if dataset.created_at else None
                    }
                    for dataset in datasets[:int(limit)]
                ]
            }
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return {"error": f"Failed to search datasets: {str(e)}"}

    async def _search_spaces(self, query: str, **filters) -> Dict[str, Any]:
        """Search for spaces with filters"""
        try:
            # Extract limit and set default
            limit = filters.pop('limit', 10)
            
            # Remove None values from filters
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            
            spaces = list(self.hf_api.list_spaces(
                search=query,
                **clean_filters
            ))
            
            return {
                "count": len(spaces),
                "results": [
                    {
                        "id": space.id,
                        "author": space.author,
                        "likes": space.likes,
                        "sdk": space.sdk,
                        "tags": space.tags[:10] if space.tags else [],  # Limit tags for brevity
                        "runtime": {
                            "stage": space.runtime.stage if space.runtime else None,
                            "hardware": space.runtime.hardware if space.runtime else None
                        } if space.runtime else None,
                        "private": space.private,
                        "created_at": space.created_at.isoformat() if space.created_at else None
                    }
                    for space in spaces[:int(limit)]
                ]
            }
        except Exception as e:
            logger.error(f"Error searching spaces: {e}")
            return {"error": f"Failed to search spaces: {str(e)}"}

    async def _get_model_details(self, model_id: str, include_files: bool = False) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            model_info = self.hf_api.model_info(
                model_id,
                files_metadata=include_files
            )
            
            result = {
                "id": model_info.id,
                "author": model_info.author,
                "sha": model_info.sha,
                "created_at": model_info.created_at.isoformat() if model_info.created_at else None,
                "last_modified": model_info.last_modified.isoformat() if model_info.last_modified else None,
                "downloads": model_info.downloads,
                "downloads_all_time": model_info.downloads_all_time if hasattr(model_info, 'downloads_all_time') else None,
                "likes": model_info.likes,
                "pipeline_tag": model_info.pipeline_tag,
                "library_name": model_info.library_name,
                "tags": model_info.tags,
                "private": model_info.private,
                "gated": model_info.gated if hasattr(model_info, 'gated') else False,
            }
            
            # Add safetensors info if available
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                result["safetensors"] = {
                    "total": model_info.safetensors.total if hasattr(model_info.safetensors, 'total') else None
                }
            
            # Add card data if available
            if hasattr(model_info, 'card_data') and model_info.card_data:
                try:
                    # Try to convert card_data to dict using to_dict() method or vars()
                    if hasattr(model_info.card_data, 'to_dict'):
                        result["card_data"] = model_info.card_data.to_dict()
                    else:
                        result["card_data"] = vars(model_info.card_data) if model_info.card_data else None
                except Exception:
                    # If conversion fails, skip card_data
                    result["card_data"] = None
            
            # Add files if requested
            if include_files and hasattr(model_info, 'siblings') and model_info.siblings:
                result["files"] = [
                    {
                        "filename": sibling.rfilename,
                        "size": getattr(sibling, 'size', None),
                    }
                    for sibling in model_info.siblings[:50]  # Limit to first 50 files
                ]
                result["total_files"] = len(model_info.siblings)
                
            return result
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            return {"error": f"Failed to get model details: {str(e)}"}

    async def _get_dataset_details(self, dataset_id: str, include_files: bool = False) -> Dict[str, Any]:
        """Get detailed dataset information"""
        try:
            dataset_info = self.hf_api.dataset_info(
                dataset_id,
                files_metadata=include_files
            )
            
            result = {
                "id": dataset_info.id,
                "author": dataset_info.author,
                "sha": dataset_info.sha,
                "created_at": dataset_info.created_at.isoformat() if dataset_info.created_at else None,
                "last_modified": dataset_info.last_modified.isoformat() if dataset_info.last_modified else None,
                "downloads": dataset_info.downloads,
                "downloads_all_time": dataset_info.downloads_all_time if hasattr(dataset_info, 'downloads_all_time') else None,
                "likes": dataset_info.likes,
                "tags": dataset_info.tags,
                "private": dataset_info.private,
                "gated": dataset_info.gated if hasattr(dataset_info, 'gated') else False,
            }
            
            # Add card data if available
            if hasattr(dataset_info, 'card_data') and dataset_info.card_data:
                try:
                    # Try to convert card_data to dict using to_dict() method or vars()
                    if hasattr(dataset_info.card_data, 'to_dict'):
                        result["card_data"] = dataset_info.card_data.to_dict()
                    else:
                        result["card_data"] = vars(dataset_info.card_data) if dataset_info.card_data else None
                except Exception:
                    # If conversion fails, skip card_data
                    result["card_data"] = None
            
            # Add files if requested
            if include_files and hasattr(dataset_info, 'siblings') and dataset_info.siblings:
                result["files"] = [
                    {
                        "filename": sibling.rfilename,
                        "size": getattr(sibling, 'size', None),
                    }
                    for sibling in dataset_info.siblings[:50]  # Limit to first 50 files
                ]
                result["total_files"] = len(dataset_info.siblings)
                
            return result
        except Exception as e:
            logger.error(f"Error getting dataset details for {dataset_id}: {e}")
            return {"error": f"Failed to get dataset details: {str(e)}"}

    async def _get_space_details(self, space_id: str, include_files: bool = False) -> Dict[str, Any]:
        """Get detailed space information"""
        try:
            space_info = self.hf_api.space_info(
                space_id,
                files_metadata=include_files
            )
            
            result = {
                "id": space_info.id,
                "author": space_info.author,
                "sha": space_info.sha,
                "created_at": space_info.created_at.isoformat() if space_info.created_at else None,
                "last_modified": space_info.last_modified.isoformat() if space_info.last_modified else None,
                "likes": space_info.likes,
                "sdk": space_info.sdk,
                "tags": space_info.tags,
                "models": space_info.models if hasattr(space_info, 'models') else [],
                "datasets": space_info.datasets if hasattr(space_info, 'datasets') else [],
                "private": space_info.private,
                "gated": space_info.gated if hasattr(space_info, 'gated') else False,
            }
            
            # Add runtime info if available
            if hasattr(space_info, 'runtime') and space_info.runtime:
                result["runtime"] = {
                    "stage": space_info.runtime.stage if hasattr(space_info.runtime, 'stage') else None,
                    "hardware": space_info.runtime.hardware if hasattr(space_info.runtime, 'hardware') else None
                }
            
            # Add card data if available
            if hasattr(space_info, 'card_data') and space_info.card_data:
                try:
                    # Try to convert card_data to dict using to_dict() method or vars()
                    if hasattr(space_info.card_data, 'to_dict'):
                        result["card_data"] = space_info.card_data.to_dict()
                    else:
                        result["card_data"] = vars(space_info.card_data) if space_info.card_data else None
                except Exception:
                    # If conversion fails, skip card_data
                    result["card_data"] = None
            
            # Add files if requested
            if include_files and hasattr(space_info, 'siblings') and space_info.siblings:
                result["files"] = [
                    {
                        "filename": sibling.rfilename,
                        "size": getattr(sibling, 'size', None),
                    }
                    for sibling in space_info.siblings[:50]  # Limit to first 50 files
                ]
                result["total_files"] = len(space_info.siblings)
                
            return result
        except Exception as e:
            logger.error(f"Error getting space details for {space_id}: {e}")
            return {"error": f"Failed to get space details: {str(e)}"}

    async def _analyze_repo_structure(self, repo_id: str, repo_type: str, path: str = None) -> Dict[str, Any]:
        """Analyze repository file structure"""
        try:
            tree_items = list(self.hf_api.list_repo_tree(
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=path,
                recursive=True
            ))
            
            files = [item for item in tree_items if hasattr(item, 'size')]
            folders = [item for item in tree_items if hasattr(item, 'tree_id')]
            
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_analyzed": path or "root",
                "total_files": len(files),
                "total_folders": len(folders),
                "total_size_bytes": sum(getattr(f, 'size', 0) for f in files),
                "file_types": self._analyze_file_types(files),
                "largest_files": sorted(
                    [{"path": f.path, "size": getattr(f, 'size', 0)} for f in files],
                    key=lambda x: x['size'],
                    reverse=True
                )[:10]
            }
        except Exception as e:
            return {"error": f"Failed to analyze repo structure: {str(e)}"}

    async def _compare_models(self, model_ids: List[str], criteria: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            models_info = []
            for model_id in model_ids:
                try:
                    model_info = self.hf_api.model_info(model_id)
                    models_info.append(model_info)
                except Exception as e:
                    logger.warning(f"Failed to fetch model {model_id}: {e}")
                    models_info.append({"id": model_id, "error": str(e)})
            
            comparison = {}
            default_criteria = ["downloads", "likes", "created_at", "library_name", "pipeline_tag"]
            criteria = criteria or default_criteria
            
            for criterion in criteria:
                if criterion == "downloads":
                    comparison[criterion] = {model.id: getattr(model, 'downloads', 0) for model in models_info if hasattr(model, 'downloads')}
                elif criterion == "likes":
                    comparison[criterion] = {model.id: getattr(model, 'likes', 0) for model in models_info if hasattr(model, 'likes')}
                elif criterion == "created_at":
                    comparison[criterion] = {
                        model.id: model.created_at.isoformat() 
                        for model in models_info 
                        if hasattr(model, 'created_at') and model.created_at
                    }
                elif criterion == "library_name":
                    comparison[criterion] = {model.id: getattr(model, 'library_name', None) for model in models_info if hasattr(model, 'library_name')}
                elif criterion == "pipeline_tag":
                    comparison[criterion] = {model.id: getattr(model, 'pipeline_tag', None) for model in models_info if hasattr(model, 'pipeline_tag')}
                    
            return {
                "models_compared": model_ids,
                "criteria_used": criteria,
                "comparison": comparison
            }
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {"error": f"Failed to compare models: {str(e)}"}

    def _analyze_file_types(self, files):
        """Analyze file types in repository"""
        from collections import Counter
        extensions = []
        for f in files:
            path = getattr(f, 'path', '')
            if '.' in path:
                ext = path.split('.')[-1].lower()
                extensions.append(ext)
        return dict(Counter(extensions))


async def main():
    """Main entry point"""
    logger.info("Starting HuggingFace Hub MCP Server...")
    server = HuggingFaceHubMCPServer()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server running and ready to accept connections")
            await server.app.run(
                read_stream,
                write_stream,
                server.app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise