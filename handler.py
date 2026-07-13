import json
import os
import time
from dotenv import load_dotenv
import random
from google import genai
from google.genai import types

load_dotenv()


scenarios = []
with open("./scenarios.txt", "r", encoding="utf-8") as s:
    lines = s.readlines()
    for line in lines:
        text = line.strip()
        scenarios.append(text)

languages = ["Português do Brasil", "English"]
prompt_base = """You are a Senior Software Architect expert in TypeScript, Zod, and RESTful API design.
Your task is to generate realistic, complex API data and strictly output its corresponding modular Zod schema using the `@asteasolutions/zod-to-openapi` library.

CRITICAL RULES:
1. SCHEMA NORMALIZATION & EXTRACTION: Never create a single giant inline schema. You must analyze the JSON structure and identify nested entities, arrays of objects, or recurring structural patterns (e.g., shared metadata, pagination objects, address blocks, or relational links). Extract these distinct patterns into their own independent `const` schemas.
2. COMPOSITION: Build the final root schema by composing it using the extracted smaller schemas (either by direct reference or using `.merge()` when inheriting flat properties).
3. OPENAPI METADATA: Every single field must have a `.openapi({ description: '...', example: ... })` method attached.
4. BILINGUAL DOCUMENTATION: JSON keys and TS variables must be in English. However, the text inside the `.openapi(description)` Write all .openapi(description) texts entirely in the requested language.
5. STRICT OUTPUT: You must respond ONLY with a valid JSON object containing exactly two keys: "raw_json" and "zod_code". Do not use markdown blocks.

"""
examples = [
    # Exemplo 1: Padrão de Paginação e Metadados (100% Inglês)
    """--- EXAMPLE ---
User Scenario: 'Paginated list of blog posts'
Output:
{
  "raw_json": {
    "data": [
      { "id": "post_1", "title": "AI Trends", "author": { "id": "usr_1", "name": "Edson" } }
    ],
    "meta": { "current_page": 1, "total_pages": 5, "has_next": true }
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst AuthorSchema = z.object({\\n  id: z.string().openapi({ description: 'Unique author identifier', example: 'usr_1' }),\\n  name: z.string().openapi({ description: 'Author full name', example: 'Edson' })\\n}).openapi({ description: 'Author details object' });\\n\\nconst BlogPostSchema = z.object({\\n  id: z.string().openapi({ description: 'Post identifier', example: 'post_1' }),\\n  title: z.string().openapi({ description: 'Title of the publication', example: 'AI Trends' }),\\n  author: AuthorSchema\\n}).openapi({ description: 'Individual blog post entity' });\\n\\nconst PaginationMetaSchema = z.object({\\n  current_page: z.number().openapi({ description: 'Current page of the results', example: 1 }),\\n  total_pages: z.number().openapi({ description: 'Total available pages', example: 5 }),\\n  has_next: z.boolean().openapi({ description: 'Indicates if there is a next page', example: true })\\n}).openapi({ description: 'Pagination metadata' });\\n\\nexport const PaginatedPostsResponseSchema = z.object({\\n  data: z.array(BlogPostSchema).openapi({ description: 'List of returned posts' }),\\n  meta: PaginationMetaSchema\\n}).openapi({ title: 'Paginated Posts', description: 'Response containing a list of posts and pagination data' });"
}""",
    # Exemplo 2: Padrão de Reuso (100% Português)
    """--- EXAMPLE ---
User Scenario: 'Customer profile with billing and shipping addresses'
Output:
{
  "raw_json": {
    "customer_id": "cust_88",
    "name": "Tech Corp",
    "billing_address": { "city": "São Paulo", "zip": "01000-000" },
    "shipping_address": { "city": "Rio de Janeiro", "zip": "20000-000" }
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst AddressSchema = z.object({\\n  city: z.string().openapi({ description: 'Nome da cidade', example: 'São Paulo' }),\\n  zip: z.string().openapi({ description: 'Código postal ou CEP', example: '01000-000' })\\n}).openapi({ description: 'Bloco de endereço padronizado' });\\n\\nexport const CustomerProfileSchema = z.object({\\n  customer_id: z.string().openapi({ description: 'Identificador único do cliente', example: 'cust_88' }),\\n  name: z.string().openapi({ description: 'Nome da empresa ou cliente', example: 'Tech Corp' }),\\n  billing_address: AddressSchema.openapi({ description: 'Endereço de cobrança' }),\\n  shipping_address: AddressSchema.openapi({ description: 'Endereço de entrega' })\\n}).openapi({ title: 'Perfil do Cliente', description: 'Dados do cliente incluindo múltiplos endereços' });"
}""",
    # Exemplo 3: Padrão HATEOAS Nível 3 (100% Inglês)
    """--- EXAMPLE ---
    
User Scenario: 'List of tasks with individual and root navigation links'
Output:
{
  "raw_json": {
    "tasks": [
      { "id": "tsk_1", "title": "Buy milk", "_links": { "self": "/tasks/tsk_1", "complete": "/tasks/tsk_1/complete" } }
    ],
    "_links": { "self": "/tasks", "create": "/tasks/new" }
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst RootLinksSchema = z.object({\\n  self: z.string().url().openapi({ description: 'Current URL of the list', example: '/tasks' }),\\n  create: z.string().url().openapi({ description: 'Endpoint to create a new task', example: '/tasks/new' })\\n}).openapi({ description: 'Root navigation links' });\\n\\nconst TaskLinksSchema = z.object({\\n  self: z.string().url().openapi({ description: 'URL of the specific task', example: '/tasks/tsk_1' }),\\n  complete: z.string().url().openapi({ description: 'Endpoint to mark the task as completed', example: '/tasks/tsk_1/complete' })\\n}).openapi({ description: 'Individual task actions' });\\n\\nconst TaskItemSchema = z.object({\\n  id: z.string().openapi({ description: 'Task ID', example: 'tsk_1' }),\\n  title: z.string().openapi({ description: 'Title of the task to be done', example: 'Buy milk' })\\n}).merge(TaskLinksSchema).openapi({ description: 'A single task entity' });\\n\\nexport const TaskListResponseSchema = z.object({\\n  tasks: z.array(TaskItemSchema).openapi({ description: 'Array of user tasks' })\\n}).merge(RootLinksSchema).openapi({ title: 'Task List', description: 'Collection of tasks with HATEOAS navigation' });"
}""",
    # Exemplo 4: Padrão de Respostas de Erro (100% Inglês)
    """--- EXAMPLE ---
User Scenario: 'Standardized API validation error response'
Output:
{
  "raw_json": {
    "error_code": "VALIDATION_FAILED",
    "message": "The request contains invalid parameters.",
    "details": [
      { "field": "email", "issue": "Invalid email format" },
      { "field": "password", "issue": "Password too short" }
    ]
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst ErrorDetailSchema = z.object({\\n  field: z.string().openapi({ description: 'The specific field that caused the error', example: 'email' }),\\n  issue: z.string().openapi({ description: 'Description of the validation issue', example: 'Invalid email format' })\\n}).openapi({ description: 'Detailed validation error for a single field' });\\n\\nexport const ErrorResponseSchema = z.object({\\n  error_code: z.string().openapi({ description: 'High-level error classification code', example: 'VALIDATION_FAILED' }),\\n  message: z.string().openapi({ description: 'Human-readable error message', example: 'The request contains invalid parameters.' }),\\n  details: z.array(ErrorDetailSchema).optional().openapi({ description: 'List of specific field validation errors' })\\n}).openapi({ title: 'Validation Error Response', description: 'Standardized structure for API errors' });"
}""",
    # Exemplo 5: Padrão de Enums e Categorização (100% Português)
    """--- EXAMPLE ---
User Scenario: 'Product details with stock status and category information'
Output:
{
  "raw_json": {
    "product_id": "prod_774",
    "status": "in_stock",
    "category": { "id": "cat_2", "name": "Eletrônicos", "department": "Tecnologia" }
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst CategorySchema = z.object({\\n  id: z.string().openapi({ description: 'Identificador da categoria', example: 'cat_2' }),\\n  name: z.string().openapi({ description: 'Nome de exibição da categoria', example: 'Eletrônicos' }),\\n  department: z.string().openapi({ description: 'Departamento principal', example: 'Tecnologia' })\\n}).openapi({ description: 'Dados de categorização do produto' });\\n\\nexport const ProductInventorySchema = z.object({\\n  product_id: z.string().openapi({ description: 'ID único do produto no sistema', example: 'prod_774' }),\\n  status: z.enum(['in_stock', 'out_of_stock', 'discontinued']).openapi({ description: 'Status atual de disponibilidade', example: 'in_stock' }),\\n  category: CategorySchema\\n}).openapi({ title: 'Detalhes do Produto', description: 'Informações de estoque e classificação de um produto' });"
}""",
    # Exemplo 6: Padrão de Agrupamento de Domínio (100% Português)
    """--- EXAMPLE ---
User Scenario: 'IoT Device health status with grouped metrics'
Output:
{
  "raw_json": {
    "device_id": "iot_cam_01",
    "uptime_seconds": 86400,
    "hardware": { "cpu_temp_celsius": 45.5, "battery_level": 88 },
    "network": { "signal_strength": "good", "ip_address": "192.168.1.10" }
  },
  "zod_code": "import { z } from 'zod';\\nimport { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';\\n\\nextendZodWithOpenApi(z);\\n\\nconst HardwareMetricsSchema = z.object({\\n  cpu_temp_celsius: z.number().openapi({ description: 'Temperatura da CPU em graus Celsius', example: 45.5 }),\\n  battery_level: z.number().min(0).max(100).openapi({ description: 'Porcentagem atual da bateria', example: 88 })\\n}).openapi({ description: 'Métricas físicas do dispositivo' });\\n\\nconst NetworkStatusSchema = z.object({\\n  signal_strength: z.enum(['poor', 'fair', 'good', 'excellent']).openapi({ description: 'Qualidade do sinal de rede', example: 'good' }),\\n  ip_address: z.string().ip().openapi({ description: 'Endereço IP local na rede', example: '192.168.1.10' })\\n}).openapi({ description: 'Status de conectividade do dispositivo' });\\n\\nexport const DeviceHealthSchema = z.object({\\n  device_id: z.string().openapi({ description: 'ID único do dispositivo IoT', example: 'iot_cam_01' }),\\n  uptime_seconds: z.number().openapi({ description: 'Tempo ligado em segundos', example: 86400 }),\\n  hardware: HardwareMetricsSchema,\\n  network: NetworkStatusSchema\\n}).openapi({ title: 'Saúde do Dispositivo', description: 'Diagnóstico completo do equipamento' });"
}""",
]

client = genai.Client()


def generateSchema(scenario):
    """Essa função coleta dados da azure para simulação"""
    language = random.choice(languages)
    example = random.choice(examples)

    print("Linguagem", language)
    config = genai.types.GenerateContentConfig(
        system_instruction=f"{prompt_base} \n CRITICAL: Write ALL openapi descriptions and examples EXCLUSIVELY in {language}. \n {example}",
        temperature=0.2,
    )
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents=scenario, config=config
    )

    return response


if __name__ == "__main__":
    i = 0
    while True:
        if i > 490:
            print("ACABOU A COTA")
            break
        print("INICIANDO")
        scenario = scenarios[i % len(scenarios)]
        try:
            result = generateSchema(scenario)
        except:
            print("\nHouve um erro, esperando mais 10 minuto\n")
            time.sleep(60 * 10)
            continue

        with open("zod.jsonl", "a", encoding="utf-8") as path:
            text = json.dumps(result.text, ensure_ascii=False)
            path.write(text + "\n")

        time.sleep(4)
        print("ENCERRANDO", i)
        i += 1
