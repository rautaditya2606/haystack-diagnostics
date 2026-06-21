import os
import pathlib
import tempfile
from typing import Any, Dict


def _get_mermaid_text(pipeline) -> str:
    """
    Leverages Haystack's native utilities to export the pipeline
    graph as a Mermaid markdown diagram. If those fail or are not
    supported in the current environment, falls back to traversing
    the pipeline.graph manually.
    """
    # 1. Try importing the internal helper from haystack to generate mermaid text directly
    try:
        from haystack.core.pipeline.draw import _to_mermaid_text
        return _to_mermaid_text(pipeline.graph, {})
    except Exception:
        pass

    # 2. Try the tempfile draw method (in case other versions support it differently)
    import os
    import pathlib
    import tempfile
    
    path_file = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        path = pathlib.Path(temp_path)
        path_file = path
        
        # Try with format argument first, then try params dict if it raises TypeError
        try:
            pipeline.draw(path, format="mermaid-text")
        except TypeError:
            pipeline.draw(path, params={"format": "mermaid-text"})
            
        return path.read_text()
    except Exception:
        pass
    finally:
        if path_file and path_file.exists():
            try:
                path_file.unlink()
            except Exception:
                pass

    # 3. Fallback: Generate Mermaid manually from pipeline.graph traversal
    try:
        graph = pipeline.graph
        if graph is None:
            return "Error: Pipeline graph is not available."
            
        lines = ["graph TD;"]
        
        # Define nodes
        for name, data in graph.nodes(data=True):
            if name in ("input", "output"):
                continue
            instance = data.get("instance")
            class_name = instance.__class__.__name__ if instance else "Component"
            lines.append(f'    {name}["<b>{name}</b><br><small><i>{class_name}</i></small>"]:::component')
            
        # Define connections
        for u, v, data in graph.edges(data=True):
            if u == "input" or v == "output":
                continue
            from_sock = data.get("from_socket")
            to_sock = data.get("to_socket")
            from_label = from_sock.name if from_sock else "output"
            to_label = to_sock.name if to_sock else "input"
            conn_type = data.get("conn_type", "")
            
            label = f'"{from_label} -> {to_label}'
            if conn_type:
                label += f' ({conn_type})'
            label += '"'
            
            lines.append(f'    {u} -->|{label}| {v}')
            
        # Add styles
        lines.append("")
        lines.append("    classDef component fill:#4A90E2,stroke:#357ABD,color:#FFFFFF,stroke-width:2px,rx:5px,ry:5px;")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error traversing pipeline graph: {str(e)}"


def inspect_pipeline(pipeline) -> Dict[str, Any]:
    """
    Formats the existing Haystack pipeline graph metadata, components,
    sockets, and connections.

    :param pipeline: The Haystack Pipeline instance.
    :return: A dictionary representing the structural metadata of the pipeline.
    """
    # Expose metadata and basics
    metadata = getattr(pipeline, "metadata", {})
    max_runs = getattr(pipeline, "_max_runs_per_component", 100)
    connection_type_validation = getattr(pipeline, "_connection_type_validation", True)

    components_details = {}
    connections_list = []

    # 1. Inspect components from the graph nodes
    if hasattr(pipeline, "graph") and pipeline.graph is not None:
        for node_name, attrs in pipeline.graph.nodes(data=True):
            instance = attrs.get("instance")
            class_name = instance.__class__.__name__ if instance else "UnknownComponent"
            module_name = instance.__class__.__module__ if instance else "unknown"
            
            # Input sockets representation
            input_sockets = []
            for socket_name, socket in attrs.get("input_sockets", {}).items():
                # Determine default value representation
                empty_sentinel = getattr(socket, "_empty", None)
                # If empty sentinel is not defined, we can try importing from haystack.core.component.types
                if empty_sentinel is None:
                    try:
                        from haystack.core.component.types import _empty
                        empty_sentinel = _empty
                    except Exception:
                        pass
                
                default_val = socket.default_value
                has_default = default_val != empty_sentinel
                
                # Check for mandatory/optional logic
                # In Haystack 2.x, a socket is mandatory if it does not have a default value
                # and is not variadic/lazy/etc.
                is_mandatory = getattr(socket, "is_mandatory", not has_default)

                input_sockets.append({
                    "name": socket_name,
                    "type": str(socket.type),
                    "is_mandatory": is_mandatory,
                    "default_value": str(default_val) if has_default else None,
                })

            # Output sockets representation
            output_sockets = []
            for socket_name, socket in attrs.get("output_sockets", {}).items():
                output_sockets.append({
                    "name": socket_name,
                    "type": str(socket.type),
                    "receivers": list(getattr(socket, "receivers", [])),
                })

            components_details[node_name] = {
                "class_name": class_name,
                "type": f"{module_name}.{class_name}",
                "input_sockets": input_sockets,
                "output_sockets": output_sockets,
            }

        # 2. Inspect connections from the graph edges
        for sender, receiver, edge_attrs in pipeline.graph.edges(data=True):
            from_socket = edge_attrs.get("from_socket")
            to_socket = edge_attrs.get("to_socket")
            connections_list.append({
                "sender": sender,
                "sender_socket": from_socket.name if from_socket else "unknown",
                "receiver": receiver,
                "receiver_socket": to_socket.name if to_socket else "unknown",
            })
    else:
        # Fallback to to_dict if graph is unavailable
        pipe_dict = pipeline.to_dict()
        for comp_name, comp_info in pipe_dict.get("components", {}).items():
            components_details[comp_name] = {
                "class_name": comp_info.get("type", "").split(".")[-1],
                "type": comp_info.get("type", ""),
                "input_sockets": [],
                "output_sockets": [],
            }
        for conn in pipe_dict.get("connections", []):
            sender_parts = conn.get("sender", "").split(".")
            receiver_parts = conn.get("receiver", "").split(".")
            connections_list.append({
                "sender": sender_parts[0] if len(sender_parts) > 0 else "",
                "sender_socket": sender_parts[1] if len(sender_parts) > 1 else "",
                "receiver": receiver_parts[0] if len(receiver_parts) > 0 else "",
                "receiver_socket": receiver_parts[1] if len(receiver_parts) > 1 else "",
            })

    # 3. Native Mermaid diagram text
    mermaid_diagram = _get_mermaid_text(pipeline)

    return {
        "metadata": metadata,
        "max_runs_per_component": max_runs,
        "connection_type_validation": connection_type_validation,
        "components": components_details,
        "connections": connections_list,
        "mermaid": mermaid_diagram,
    }
