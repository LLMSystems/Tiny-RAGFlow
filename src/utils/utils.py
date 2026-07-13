import re

def _normalize_json_response(response: str) -> str:
        if not response:
            return "{}"
        
        response_clean = response.strip()
        
        json_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`json\s*(.*?)\s*`',
            r'`\s*(.*?)\s*`'
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, response_clean, re.DOTALL)
            if match:
                response_clean = match.group(1).strip()
                break
        
        start_idx = response_clean.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_clean[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count == 0:
                response_clean = response_clean[start_idx:end_idx + 1]
        
        response_clean = re.sub(r"'([^']*)':", r'"\1":', response_clean)
        response_clean = re.sub(r":\s*'([^']*)'", r': "\1"', response_clean)
        response_clean = re.sub(r',\s*}', '}', response_clean)
        response_clean = re.sub(r',\s*]', ']', response_clean)
        
        return response_clean