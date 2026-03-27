# Reference
## AnnotationQueues
<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">list_queues</a>(...) -> PaginatedAnnotationQueues</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all annotation queues
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.list_queues()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">create_queue</a>(...) -> AnnotationQueue</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.create_queue(
    name="name",
    score_config_ids=[
        "scoreConfigIds",
        "scoreConfigIds"
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateAnnotationQueueRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">get_queue</a>(...) -> AnnotationQueue</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get an annotation queue by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.get_queue(
    queue_id="queueId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">list_queue_items</a>(...) -> PaginatedAnnotationQueueItems</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get items for a specific annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.list_queue_items(
    queue_id="queueId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[AnnotationQueueStatus]` — Filter by status
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">get_queue_item</a>(...) -> AnnotationQueueItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a specific item from an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.get_queue_item(
    queue_id="queueId",
    item_id="itemId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**item_id:** `str` — The unique identifier of the annotation queue item
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">create_queue_item</a>(...) -> AnnotationQueueItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Add an item to an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.create_queue_item(
    queue_id="queueId",
    object_id="objectId",
    object_type="TRACE",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**request:** `CreateAnnotationQueueItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">update_queue_item</a>(...) -> AnnotationQueueItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update an annotation queue item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.update_queue_item(
    queue_id="queueId",
    item_id="itemId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**item_id:** `str` — The unique identifier of the annotation queue item
    
</dd>
</dl>

<dl>
<dd>

**request:** `UpdateAnnotationQueueItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">delete_queue_item</a>(...) -> DeleteAnnotationQueueItemResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove an item from an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.delete_queue_item(
    queue_id="queueId",
    item_id="itemId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**item_id:** `str` — The unique identifier of the annotation queue item
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">create_queue_assignment</a>(...) -> CreateAnnotationQueueAssignmentResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create an assignment for a user to an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.create_queue_assignment(
    queue_id="queueId",
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**request:** `AnnotationQueueAssignmentRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.annotation_queues.<a href="src/langfuse/annotation_queues/client.py">delete_queue_assignment</a>(...) -> DeleteAnnotationQueueAssignmentResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete an assignment for a user to an annotation queue
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.annotation_queues.delete_queue_assignment(
    queue_id="queueId",
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**queue_id:** `str` — The unique identifier of the annotation queue
    
</dd>
</dl>

<dl>
<dd>

**request:** `AnnotationQueueAssignmentRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## BlobStorageIntegrations
<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/blob_storage_integrations/client.py">get_blob_storage_integrations</a>() -> BlobStorageIntegrationsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all blob storage integrations for the organization (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.blob_storage_integrations.get_blob_storage_integrations()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/blob_storage_integrations/client.py">upsert_blob_storage_integration</a>(...) -> BlobStorageIntegrationResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create or update a blob storage integration for a specific project (requires organization-scoped API key). The configuration is validated by performing a test upload to the bucket.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.blob_storage_integrations.upsert_blob_storage_integration(
    project_id="projectId",
    type="S3",
    bucket_name="bucketName",
    region="region",
    export_frequency="hourly",
    enabled=True,
    force_path_style=True,
    file_type="JSON",
    export_mode="FULL_HISTORY",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateBlobStorageIntegrationRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/blob_storage_integrations/client.py">get_blob_storage_integration_status</a>(...) -> BlobStorageIntegrationStatusResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get the sync status of a blob storage integration by integration ID (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.blob_storage_integrations.get_blob_storage_integration_status(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/blob_storage_integrations/client.py">delete_blob_storage_integration</a>(...) -> BlobStorageIntegrationDeletionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a blob storage integration by ID (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.blob_storage_integrations.delete_blob_storage_integration(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Comments
<details><summary><code>client.comments.<a href="src/langfuse/comments/client.py">create</a>(...) -> CreateCommentResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a comment. Comments may be attached to different object types (trace, observation, session, prompt).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.comments.create(
    project_id="projectId",
    object_type="objectType",
    object_id="objectId",
    content="content",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateCommentRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.comments.<a href="src/langfuse/comments/client.py">get</a>(...) -> GetCommentsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all comments
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.comments.get()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit
    
</dd>
</dl>

<dl>
<dd>

**object_type:** `typing.Optional[str]` — Filter comments by object type (trace, observation, session, prompt).
    
</dd>
</dl>

<dl>
<dd>

**object_id:** `typing.Optional[str]` — Filter comments by object id. If objectType is not provided, an error will be thrown.
    
</dd>
</dl>

<dl>
<dd>

**author_user_id:** `typing.Optional[str]` — Filter comments by author user id.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.comments.<a href="src/langfuse/comments/client.py">get_by_id</a>(...) -> Comment</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a comment by id
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.comments.get_by_id(
    comment_id="commentId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**comment_id:** `str` — The unique langfuse identifier of a comment
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## DatasetItems
<details><summary><code>client.dataset_items.<a href="src/langfuse/dataset_items/client.py">create</a>(...) -> DatasetItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_items.create(
    dataset_name="datasetName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_items.<a href="src/langfuse/dataset_items/client.py">get</a>(...) -> DatasetItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_items.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_items.<a href="src/langfuse/dataset_items/client.py">list</a>(...) -> PaginatedDatasetItems</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get dataset items. Optionally specify a version to get the items as they existed at that point in time.
Note: If version parameter is provided, datasetName must also be provided.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_items.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**source_trace_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**source_observation_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[datetime.datetime]` 

ISO 8601 timestamp (RFC 3339, Section 5.6) in UTC (e.g., "2026-01-21T14:35:42Z").
If provided, returns state of dataset at this timestamp.
If not provided, returns the latest version. Requires datasetName to be specified.
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_items.<a href="src/langfuse/dataset_items/client.py">delete</a>(...) -> DeleteDatasetItemResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a dataset item and all its run items. This action is irreversible.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_items.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## DatasetRunItems
<details><summary><code>client.dataset_run_items.<a href="src/langfuse/dataset_run_items/client.py">create</a>(...) -> DatasetRunItem</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset run item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_run_items.create(
    run_name="runName",
    dataset_item_id="datasetItemId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetRunItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_run_items.<a href="src/langfuse/dataset_run_items/client.py">list</a>(...) -> PaginatedDatasetRunItems</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List dataset run items
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.dataset_run_items.list(
    dataset_id="datasetId",
    run_name="runName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**run_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Datasets
<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">list</a>(...) -> PaginatedDatasets</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all datasets
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">get</a>(...) -> Dataset</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.get(
    dataset_name="datasetName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">create</a>(...) -> Dataset</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.create(
    name="name",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">get_run</a>(...) -> DatasetRunWithItems</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset run and its items
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.get_run(
    dataset_name="datasetName",
    run_name="runName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**run_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">delete_run</a>(...) -> DeleteDatasetRunResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a dataset run and all its run items. This action is irreversible.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.delete_run(
    dataset_name="datasetName",
    run_name="runName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**run_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/datasets/client.py">get_runs</a>(...) -> PaginatedDatasetRuns</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get dataset runs
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.datasets.get_runs(
    dataset_name="datasetName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Health
<details><summary><code>client.health.<a href="src/langfuse/health/client.py">health</a>() -> HealthResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Check health of API and database
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.health.health()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Ingestion
<details><summary><code>client.ingestion.<a href="src/langfuse/ingestion/client.py">batch</a>(...) -> IngestionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

**Legacy endpoint for batch ingestion for Langfuse Observability.**

-> Please use the OpenTelemetry endpoint (`/api/public/otel/v1/traces`). Learn more: https://langfuse.com/integrations/native/opentelemetry

Within each batch, there can be multiple events.
Each event has a type, an id, a timestamp, metadata and a body.
Internally, we refer to this as the "event envelope" as it tells us something about the event but not the trace.
We use the event id within this envelope to deduplicate messages to avoid processing the same event twice, i.e. the event id should be unique per request.
The event.body.id is the ID of the actual trace and will be used for updates and will be visible within the Langfuse App.
I.e. if you want to update a trace, you'd use the same body id, but separate event IDs.

Notes:
- Introduction to data model: https://langfuse.com/docs/observability/data-model
- Batch sizes are limited to 3.5 MB in total. You need to adjust the number of events per batch accordingly.
- The API does not return a 4xx status code for input errors. Instead, it responds with a 207 status code, which includes a list of the encountered errors.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI
from langfuse.ingestion import TraceBody
import datetime

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.ingestion.batch(
    batch=[
        {
            "type": "trace-create",
            "id": "abcdef-1234-5678-90ab",
            "timestamp": "2022-01-01T00:00:00.000Z",
            "body": TraceBody(
                id="abcdef-1234-5678-90ab",
                timestamp=datetime.datetime.fromisoformat("2022-01-01T00:00:00.000+00:00"),
                environment="production",
                name="My Trace",
                user_id="1234-5678-90ab-cdef",
                input="My input",
                output="My output",
                session_id="1234-5678-90ab-cdef",
                release="1.0.0",
                version="1.0.0",
                metadata="My metadata",
                tags=[
                    "tag1",
                    "tag2"
                ],
                public=True,
            )
        }
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**batch:** `typing.List[IngestionEvent]` — Batch of tracing events to be ingested. Discriminated by attribute `type`.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Any]` — Optional. Metadata field used by the Langfuse SDKs for debugging.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Legacy MetricsV1
<details><summary><code>client.legacy.metrics_v1.<a href="src/langfuse/legacy/metrics_v1/client.py">metrics</a>(...) -> MetricsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get metrics from the Langfuse project using a query object.

Consider using the [v2 metrics endpoint](/api-reference#tag/metricsv2/GET/api/public/v2/metrics) for better performance.

For more details, see the [Metrics API documentation](https://langfuse.com/docs/metrics/features/metrics-api).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.legacy.metrics_v1.metrics(
    query="query",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**query:** `str` 

JSON string containing the query parameters with the following structure:
```json
{
  "view": string,           // Required. One of "traces", "observations", "scores-numeric", "scores-categorical"
  "dimensions": [           // Optional. Default: []
    {
      "field": string       // Field to group by, e.g. "name", "userId", "sessionId"
    }
  ],
  "metrics": [              // Required. At least one metric must be provided
    {
      "measure": string,    // What to measure, e.g. "count", "latency", "value"
      "aggregation": string // How to aggregate, e.g. "count", "sum", "avg", "p95", "histogram"
    }
  ],
  "filters": [              // Optional. Default: []
    {
      "column": string,     // Column to filter on
      "operator": string,   // Operator, e.g. "=", ">", "<", "contains"
      "value": any,         // Value to compare against
      "type": string,       // Data type, e.g. "string", "number", "stringObject"
      "key": string         // Required only when filtering on metadata
    }
  ],
  "timeDimension": {        // Optional. Default: null. If provided, results will be grouped by time
    "granularity": string   // One of "minute", "hour", "day", "week", "month", "auto"
  },
  "fromTimestamp": string,  // Required. ISO datetime string for start of time range
  "toTimestamp": string,    // Required. ISO datetime string for end of time range
  "orderBy": [              // Optional. Default: null
    {
      "field": string,      // Field to order by
      "direction": string   // "asc" or "desc"
    }
  ],
  "config": {               // Optional. Query-specific configuration
    "bins": number,         // Optional. Number of bins for histogram (1-100), default: 10
    "row_limit": number     // Optional. Row limit for results (1-1000)
  }
}
```
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Legacy ObservationsV1
<details><summary><code>client.legacy.observations_v1.<a href="src/langfuse/legacy/observations_v1/client.py">get</a>(...) -> ObservationsView</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a observation
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.legacy.observations_v1.get(
    observation_id="observationId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**observation_id:** `str` — The unique langfuse identifier of an observation, can be an event, span or generation
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.legacy.observations_v1.<a href="src/langfuse/legacy/observations_v1/client.py">get_many</a>(...) -> ObservationsViews</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of observations.

Consider using the [v2 observations endpoint](/api-reference#tag/observationsv2/GET/api/public/v2/observations) for cursor-based pagination and field selection.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.legacy.observations_v1.get_many()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**type:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**trace_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**level:** `typing.Optional[ObservationLevel]` — Optional filter for observations with a specific level (e.g. "DEBUG", "DEFAULT", "WARNING", "ERROR").
    
</dd>
</dl>

<dl>
<dd>

**parent_observation_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for observations where the environment is one of the provided values.
    
</dd>
</dl>

<dl>
<dd>

**from_start_time:** `typing.Optional[datetime.datetime]` — Retrieve only observations with a start_time on or after this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**to_start_time:** `typing.Optional[datetime.datetime]` — Retrieve only observations with a start_time before this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[str]` — Optional filter to only include observations with a certain version.
    
</dd>
</dl>

<dl>
<dd>

**filter:** `typing.Optional[str]` 

JSON string containing an array of filter conditions. When provided, this takes precedence over query parameter filters (userId, name, type, level, environment, fromStartTime, ...).

## Filter Structure
Each filter condition has the following structure:
```json
[
  {
    "type": string,           // Required. One of: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
    "column": string,         // Required. Column to filter on (see available columns below)
    "operator": string,       // Required. Operator based on type:
                              // - datetime: ">", "<", ">=", "<="
                              // - string: "=", "contains", "does not contain", "starts with", "ends with"
                              // - stringOptions: "any of", "none of"
                              // - categoryOptions: "any of", "none of"
                              // - arrayOptions: "any of", "none of", "all of"
                              // - number: "=", ">", "<", ">=", "<="
                              // - stringObject: "=", "contains", "does not contain", "starts with", "ends with"
                              // - numberObject: "=", ">", "<", ">=", "<="
                              // - boolean: "=", "<>"
                              // - null: "is null", "is not null"
    "value": any,             // Required (except for null type). Value to compare against. Type depends on filter type
    "key": string             // Required only for stringObject, numberObject, and categoryOptions types when filtering on nested fields like metadata
  }
]
```

## Available Columns

### Core Observation Fields
- `id` (string) - Observation ID
- `type` (string) - Observation type (SPAN, GENERATION, EVENT)
- `name` (string) - Observation name
- `traceId` (string) - Associated trace ID
- `startTime` (datetime) - Observation start time
- `endTime` (datetime) - Observation end time
- `environment` (string) - Environment tag
- `level` (string) - Log level (DEBUG, DEFAULT, WARNING, ERROR)
- `statusMessage` (string) - Status message
- `version` (string) - Version tag

### Performance Metrics
- `latency` (number) - Latency in seconds (calculated: end_time - start_time)
- `timeToFirstToken` (number) - Time to first token in seconds
- `tokensPerSecond` (number) - Output tokens per second

### Token Usage
- `inputTokens` (number) - Number of input tokens
- `outputTokens` (number) - Number of output tokens
- `totalTokens` (number) - Total tokens (alias: `tokens`)

### Cost Metrics
- `inputCost` (number) - Input cost in USD
- `outputCost` (number) - Output cost in USD
- `totalCost` (number) - Total cost in USD

### Model Information
- `model` (string) - Provided model name
- `promptName` (string) - Associated prompt name
- `promptVersion` (number) - Associated prompt version

### Structured Data
- `metadata` (stringObject/numberObject/categoryOptions) - Metadata key-value pairs. Use `key` parameter to filter on specific metadata keys.

### Associated Trace Fields (requires join with traces table)
- `userId` (string) - User ID from associated trace
- `traceName` (string) - Name from associated trace
- `traceEnvironment` (string) - Environment from associated trace
- `traceTags` (arrayOptions) - Tags from associated trace

## Filter Examples
```json
[
  {
    "type": "string",
    "column": "type",
    "operator": "=",
    "value": "GENERATION"
  },
  {
    "type": "number",
    "column": "latency",
    "operator": ">=",
    "value": 2.5
  },
  {
    "type": "stringObject",
    "column": "metadata",
    "key": "environment",
    "operator": "=",
    "value": "production"
  }
]
```
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Legacy ScoreV1
<details><summary><code>client.legacy.score_v1.<a href="src/langfuse/legacy/score_v1/client.py">create</a>(...) -> CreateScoreResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a score (supports both trace and session scores)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.legacy.score_v1.create(
    name="name",
    value=1.1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateScoreRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.legacy.score_v1.<a href="src/langfuse/legacy/score_v1/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a score (supports both trace and session scores)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.legacy.score_v1.delete(
    score_id="scoreId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**score_id:** `str` — The unique langfuse identifier of a score
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## LlmConnections
<details><summary><code>client.llm_connections.<a href="src/langfuse/llm_connections/client.py">list</a>(...) -> PaginatedLlmConnections</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all LLM connections in a project
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.llm_connections.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.llm_connections.<a href="src/langfuse/llm_connections/client.py">upsert</a>(...) -> LlmConnection</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create or update an LLM connection. The connection is upserted on provider.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.llm_connections.upsert(
    provider="provider",
    adapter="anthropic",
    secret_key="secretKey",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `UpsertLlmConnectionRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Media
<details><summary><code>client.media.<a href="src/langfuse/media/client.py">get</a>(...) -> GetMediaResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.media.get(
    media_id="mediaId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**media_id:** `str` — The unique langfuse identifier of a media record
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.media.<a href="src/langfuse/media/client.py">patch</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Patch a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI
import datetime

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.media.patch(
    media_id="mediaId",
    uploaded_at=datetime.datetime.fromisoformat("2024-01-15T09:30:00+00:00"),
    upload_http_status=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**media_id:** `str` — The unique langfuse identifier of a media record
    
</dd>
</dl>

<dl>
<dd>

**request:** `PatchMediaBody` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.media.<a href="src/langfuse/media/client.py">get_upload_url</a>(...) -> GetMediaUploadUrlResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a presigned upload URL for a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.media.get_upload_url(
    trace_id="traceId",
    content_type="image/png",
    content_length=1,
    sha256hash="sha256Hash",
    field="field",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `GetMediaUploadUrlRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Metrics
<details><summary><code>client.metrics.<a href="src/langfuse/metrics/client.py">metrics</a>(...) -> MetricsV2Response</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get metrics from the Langfuse project using a query object. V2 endpoint with optimized performance.

## V2 Differences
- Supports `observations`, `scores-numeric`, and `scores-categorical` views only (traces view not supported)
- Direct access to tags and release fields on observations
- Backwards-compatible: traceName, traceRelease, traceVersion dimensions are still available on observations view
- High cardinality dimensions are not supported and will return a 400 error (see below)

For more details, see the [Metrics API documentation](https://langfuse.com/docs/metrics/features/metrics-api).

## Available Views

### observations
Query observation-level data (spans, generations, events).

**Dimensions:**
- `environment` - Deployment environment (e.g., production, staging)
- `type` - Type of observation (SPAN, GENERATION, EVENT)
- `name` - Name of the observation
- `level` - Logging level of the observation
- `version` - Version of the observation
- `tags` - User-defined tags
- `release` - Release version
- `traceName` - Name of the parent trace (backwards-compatible)
- `traceRelease` - Release version of the parent trace (backwards-compatible, maps to release)
- `traceVersion` - Version of the parent trace (backwards-compatible, maps to version)
- `providedModelName` - Name of the model used
- `promptName` - Name of the prompt used
- `promptVersion` - Version of the prompt used
- `startTimeMonth` - Month of start_time in YYYY-MM format

**Measures:**
- `count` - Total number of observations
- `latency` - Observation latency (milliseconds)
- `streamingLatency` - Generation latency from completion start to end (milliseconds)
- `inputTokens` - Sum of input tokens consumed
- `outputTokens` - Sum of output tokens produced
- `totalTokens` - Sum of all tokens consumed
- `outputTokensPerSecond` - Output tokens per second
- `tokensPerSecond` - Total tokens per second
- `inputCost` - Input cost (USD)
- `outputCost` - Output cost (USD)
- `totalCost` - Total cost (USD)
- `timeToFirstToken` - Time to first token (milliseconds)
- `countScores` - Number of scores attached to the observation

### scores-numeric
Query numeric and boolean score data.

**Dimensions:**
- `environment` - Deployment environment
- `name` - Name of the score (e.g., accuracy, toxicity)
- `source` - Origin of the score (API, ANNOTATION, EVAL)
- `dataType` - Data type (NUMERIC, BOOLEAN)
- `configId` - Identifier of the score config
- `timestampMonth` - Month in YYYY-MM format
- `timestampDay` - Day in YYYY-MM-DD format
- `value` - Numeric value of the score
- `traceName` - Name of the parent trace
- `tags` - Tags
- `traceRelease` - Release version
- `traceVersion` - Version
- `observationName` - Name of the associated observation
- `observationModelName` - Model name of the associated observation
- `observationPromptName` - Prompt name of the associated observation
- `observationPromptVersion` - Prompt version of the associated observation

**Measures:**
- `count` - Total number of scores
- `value` - Score value (for aggregations)

### scores-categorical
Query categorical score data. Same dimensions as scores-numeric except uses `stringValue` instead of `value`.

**Measures:**
- `count` - Total number of scores

## High Cardinality Dimensions
The following dimensions cannot be used as grouping dimensions in v2 metrics API as they can cause performance issues.
Use them in filters instead.

**observations view:**
- `id` - Use traceId filter to narrow down results
- `traceId` - Use traceId filter instead
- `userId` - Use userId filter instead
- `sessionId` - Use sessionId filter instead
- `parentObservationId` - Use parentObservationId filter instead

**scores-numeric / scores-categorical views:**
- `id` - Use specific filters to narrow down results
- `traceId` - Use traceId filter instead
- `userId` - Use userId filter instead
- `sessionId` - Use sessionId filter instead
- `observationId` - Use observationId filter instead

## Aggregations
Available aggregation functions: `sum`, `avg`, `count`, `max`, `min`, `p50`, `p75`, `p90`, `p95`, `p99`, `histogram`

## Time Granularities
Available granularities for timeDimension: `auto`, `minute`, `hour`, `day`, `week`, `month`
- `auto` bins the data into approximately 50 buckets based on the time range
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.metrics.metrics(
    query="query",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**query:** `str` 

JSON string containing the query parameters with the following structure:
```json
{
  "view": string,           // Required. One of "observations", "scores-numeric", "scores-categorical"
  "dimensions": [           // Optional. Default: []
    {
      "field": string       // Field to group by (see available dimensions above)
    }
  ],
  "metrics": [              // Required. At least one metric must be provided
    {
      "measure": string,    // What to measure (see available measures above)
      "aggregation": string // How to aggregate: "sum", "avg", "count", "max", "min", "p50", "p75", "p90", "p95", "p99", "histogram"
    }
  ],
  "filters": [              // Optional. Default: []
    {
      "column": string,     // Column to filter on (any dimension field)
      "operator": string,   // Operator based on type:
                            // - datetime: ">", "<", ">=", "<="
                            // - string: "=", "contains", "does not contain", "starts with", "ends with"
                            // - stringOptions: "any of", "none of"
                            // - arrayOptions: "any of", "none of", "all of"
                            // - number: "=", ">", "<", ">=", "<="
                            // - stringObject/numberObject: same as string/number with required "key"
                            // - boolean: "=", "<>"
                            // - null: "is null", "is not null"
      "value": any,         // Value to compare against
      "type": string,       // Data type: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
      "key": string         // Required only for stringObject/numberObject types (e.g., metadata filtering)
    }
  ],
  "timeDimension": {        // Optional. Default: null. If provided, results will be grouped by time
    "granularity": string   // One of "auto", "minute", "hour", "day", "week", "month"
  },
  "fromTimestamp": string,  // Required. ISO datetime string for start of time range
  "toTimestamp": string,    // Required. ISO datetime string for end of time range (must be after fromTimestamp)
  "orderBy": [              // Optional. Default: null
    {
      "field": string,      // Field to order by (dimension or metric alias)
      "direction": string   // "asc" or "desc"
    }
  ],
  "config": {               // Optional. Query-specific configuration
    "bins": number,         // Optional. Number of bins for histogram aggregation (1-100), default: 10
    "row_limit": number     // Optional. Maximum number of rows to return (1-1000), default: 100
  }
}
```
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Models
<details><summary><code>client.models.<a href="src/langfuse/models/client.py">create</a>(...) -> Model</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a model
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.models.create(
    model_name="modelName",
    match_pattern="matchPattern",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateModelRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/models/client.py">list</a>(...) -> PaginatedModels</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all models
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.models.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/models/client.py">get</a>(...) -> Model</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a model
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.models.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/models/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a model. Cannot delete models managed by Langfuse. You can create your own definition with the same modelName to override the definition though.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.models.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Observations
<details><summary><code>client.observations.<a href="src/langfuse/observations/client.py">get_many</a>(...) -> ObservationsV2Response</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of observations with cursor-based pagination and flexible field selection.

## Cursor-based Pagination
This endpoint uses cursor-based pagination for efficient traversal of large datasets.
The cursor is returned in the response metadata and should be passed in subsequent requests
to retrieve the next page of results.

## Field Selection
Use the `fields` parameter to control which observation fields are returned:
- `core` - Always included: id, traceId, startTime, endTime, projectId, parentObservationId, type
- `basic` - name, level, statusMessage, version, environment, bookmarked, public, userId, sessionId
- `time` - completionStartTime, createdAt, updatedAt
- `io` - input, output
- `metadata` - metadata (truncated to 200 chars by default, use `expandMetadata` to get full values)
- `model` - providedModelName, internalModelId, modelParameters
- `usage` - usageDetails, costDetails, totalCost
- `prompt` - promptId, promptName, promptVersion
- `metrics` - latency, timeToFirstToken

If not specified, `core` and `basic` field groups are returned.

## Filters
Multiple filtering options are available via query parameters or the structured `filter` parameter.
When using the `filter` parameter, it takes precedence over individual query parameter filters.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.observations.get_many()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**fields:** `typing.Optional[str]` 

Comma-separated list of field groups to include in the response.
Available groups: core, basic, time, io, metadata, model, usage, prompt, metrics.
If not specified, `core` and `basic` field groups are returned.
Example: "basic,usage,model"
    
</dd>
</dl>

<dl>
<dd>

**expand_metadata:** `typing.Optional[str]` 

Comma-separated list of metadata keys to return non-truncated.
By default, metadata values over 200 characters are truncated.
Use this parameter to retrieve full values for specific keys.
Example: "key1,key2"
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Number of items to return per page. Maximum 1000, default 50.
    
</dd>
</dl>

<dl>
<dd>

**cursor:** `typing.Optional[str]` — Base64-encoded cursor for pagination. Use the cursor from the previous response to get the next page.
    
</dd>
</dl>

<dl>
<dd>

**parse_io_as_json:** `typing.Optional[bool]` 

**Deprecated.** Setting this to `true` will return a 400 error.
Input/output fields are always returned as raw strings.
Remove this parameter or set it to `false`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**type:** `typing.Optional[str]` — Filter by observation type (e.g., "GENERATION", "SPAN", "EVENT", "AGENT", "TOOL", "CHAIN", "RETRIEVER", "EVALUATOR", "EMBEDDING", "GUARDRAIL")
    
</dd>
</dl>

<dl>
<dd>

**trace_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**level:** `typing.Optional[ObservationLevel]` — Optional filter for observations with a specific level (e.g. "DEBUG", "DEFAULT", "WARNING", "ERROR").
    
</dd>
</dl>

<dl>
<dd>

**parent_observation_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for observations where the environment is one of the provided values.
    
</dd>
</dl>

<dl>
<dd>

**from_start_time:** `typing.Optional[datetime.datetime]` — Retrieve only observations with a start_time on or after this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**to_start_time:** `typing.Optional[datetime.datetime]` — Retrieve only observations with a start_time before this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[str]` — Optional filter to only include observations with a certain version.
    
</dd>
</dl>

<dl>
<dd>

**filter:** `typing.Optional[str]` 

JSON string containing an array of filter conditions. When provided, this takes precedence over query parameter filters (userId, name, type, level, environment, fromStartTime, ...).

## Filter Structure
Each filter condition has the following structure:
```json
[
  {
    "type": string,           // Required. One of: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
    "column": string,         // Required. Column to filter on (see available columns below)
    "operator": string,       // Required. Operator based on type:
                              // - datetime: ">", "<", ">=", "<="
                              // - string: "=", "contains", "does not contain", "starts with", "ends with"
                              // - stringOptions: "any of", "none of"
                              // - categoryOptions: "any of", "none of"
                              // - arrayOptions: "any of", "none of", "all of"
                              // - number: "=", ">", "<", ">=", "<="
                              // - stringObject: "=", "contains", "does not contain", "starts with", "ends with"
                              // - numberObject: "=", ">", "<", ">=", "<="
                              // - boolean: "=", "<>"
                              // - null: "is null", "is not null"
    "value": any,             // Required (except for null type). Value to compare against. Type depends on filter type
    "key": string             // Required only for stringObject, numberObject, and categoryOptions types when filtering on nested fields like metadata
  }
]
```

## Available Columns

### Core Observation Fields
- `id` (string) - Observation ID
- `type` (string) - Observation type (SPAN, GENERATION, EVENT)
- `name` (string) - Observation name
- `traceId` (string) - Associated trace ID
- `startTime` (datetime) - Observation start time
- `endTime` (datetime) - Observation end time
- `environment` (string) - Environment tag
- `level` (string) - Log level (DEBUG, DEFAULT, WARNING, ERROR)
- `statusMessage` (string) - Status message
- `version` (string) - Version tag
- `userId` (string) - User ID
- `sessionId` (string) - Session ID

### Trace-Related Fields
- `traceName` (string) - Name of the parent trace
- `traceTags` (arrayOptions) - Tags from the parent trace
- `tags` (arrayOptions) - Alias for traceTags

### Performance Metrics
- `latency` (number) - Latency in seconds (calculated: end_time - start_time)
- `timeToFirstToken` (number) - Time to first token in seconds
- `tokensPerSecond` (number) - Output tokens per second

### Token Usage
- `inputTokens` (number) - Number of input tokens
- `outputTokens` (number) - Number of output tokens
- `totalTokens` (number) - Total tokens (alias: `tokens`)

### Cost Metrics
- `inputCost` (number) - Input cost in USD
- `outputCost` (number) - Output cost in USD
- `totalCost` (number) - Total cost in USD

### Model Information
- `model` (string) - Provided model name (alias: `providedModelName`)
- `promptName` (string) - Associated prompt name
- `promptVersion` (number) - Associated prompt version

### Structured Data
- `metadata` (stringObject/numberObject/categoryOptions) - Metadata key-value pairs. Use `key` parameter to filter on specific metadata keys.

## Filter Examples
```json
[
  {
    "type": "string",
    "column": "type",
    "operator": "=",
    "value": "GENERATION"
  },
  {
    "type": "number",
    "column": "latency",
    "operator": ">=",
    "value": 2.5
  },
  {
    "type": "stringObject",
    "column": "metadata",
    "key": "environment",
    "operator": "=",
    "value": "production"
  }
]
```
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Opentelemetry
<details><summary><code>client.opentelemetry.<a href="src/langfuse/opentelemetry/client.py">export_traces</a>(...) -> OtelTraceResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

**OpenTelemetry Traces Ingestion Endpoint**

This endpoint implements the OTLP/HTTP specification for trace ingestion, providing native OpenTelemetry integration for Langfuse Observability.

**Supported Formats:**
- Binary Protobuf: `Content-Type: application/x-protobuf`
- JSON Protobuf: `Content-Type: application/json`
- Supports gzip compression via `Content-Encoding: gzip` header

**Specification Compliance:**
- Conforms to [OTLP/HTTP Trace Export](https://opentelemetry.io/docs/specs/otlp/#otlphttp)
- Implements `ExportTraceServiceRequest` message format

**Documentation:**
- Integration guide: https://langfuse.com/integrations/native/opentelemetry
- Data model: https://langfuse.com/docs/observability/data-model
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI
from langfuse.opentelemetry import OtelResourceSpan, OtelResource, OtelAttribute, OtelAttributeValue, OtelScopeSpan, OtelScope, OtelSpan

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.opentelemetry.export_traces(
    resource_spans=[
        OtelResourceSpan(
            resource=OtelResource(
                attributes=[
                    OtelAttribute(
                        key="service.name",
                        value=OtelAttributeValue(
                            string_value="my-service",
                        ),
                    ),
                    OtelAttribute(
                        key="service.version",
                        value=OtelAttributeValue(
                            string_value="1.0.0",
                        ),
                    )
                ],
            ),
            scope_spans=[
                OtelScopeSpan(
                    scope=OtelScope(
                        name="langfuse-sdk",
                        version="2.60.3",
                    ),
                    spans=[
                        OtelSpan(
                            trace_id="0123456789abcdef0123456789abcdef",
                            span_id="0123456789abcdef",
                            name="my-operation",
                            kind=1,
                            start_time_unix_nano="1747872000000000000",
                            end_time_unix_nano="1747872001000000000",
                            attributes=[
                                OtelAttribute(
                                    key="langfuse.observation.type",
                                    value=OtelAttributeValue(
                                        string_value="generation",
                                    ),
                                )
                            ],
                            status={},
                        )
                    ],
                )
            ],
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**resource_spans:** `typing.List[OtelResourceSpan]` — Array of resource spans containing trace data as defined in the OTLP specification
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Organizations
<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">get_organization_memberships</a>() -> MembershipsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all memberships for the organization associated with the API key (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.get_organization_memberships()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">update_organization_membership</a>(...) -> MembershipResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create or update a membership for the organization associated with the API key (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.update_organization_membership(
    user_id="userId",
    role="OWNER",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `MembershipRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">delete_organization_membership</a>(...) -> MembershipDeletionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a membership from the organization associated with the API key (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.delete_organization_membership(
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `DeleteMembershipRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">get_project_memberships</a>(...) -> MembershipsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all memberships for a specific project (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.get_project_memberships(
    project_id="projectId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">update_project_membership</a>(...) -> MembershipResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create or update a membership for a specific project (requires organization-scoped API key). The user must already be a member of the organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.update_project_membership(
    project_id="projectId",
    user_id="userId",
    role="OWNER",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request:** `MembershipRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">delete_project_membership</a>(...) -> MembershipDeletionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a membership from a specific project (requires organization-scoped API key). The user must be a member of the organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.delete_project_membership(
    project_id="projectId",
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request:** `DeleteMembershipRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">get_organization_projects</a>() -> OrganizationProjectsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all projects for the organization associated with the API key (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.get_organization_projects()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.organizations.<a href="src/langfuse/organizations/client.py">get_organization_api_keys</a>() -> OrganizationApiKeysResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all API keys for the organization associated with the API key (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.organizations.get_organization_api_keys()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Projects
<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">get</a>() -> Projects</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get Project associated with API key (requires project-scoped API key). You can use GET /api/public/organizations/projects to get all projects with an organization-scoped key.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.get()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">create</a>(...) -> Project</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new project (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.create(
    name="name",
    retention=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**retention:** `int` — Number of days to retain data. Must be 0 or at least 3 days. Requires data-retention entitlement for non-zero values. Optional.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Optional metadata for the project
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">update</a>(...) -> Project</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a project by ID (requires organization-scoped API key).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.update(
    project_id="projectId",
    name="name",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Optional metadata for the project
    
</dd>
</dl>

<dl>
<dd>

**retention:** `typing.Optional[int]` 

Number of days to retain data.
Must be 0 or at least 3 days.
Requires data-retention entitlement for non-zero values.
Optional. Will retain existing retention setting if omitted.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">delete</a>(...) -> ProjectDeletionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a project by ID (requires organization-scoped API key). Project deletion is processed asynchronously.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.delete(
    project_id="projectId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">get_api_keys</a>(...) -> ApiKeyList</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all API keys for a project (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.get_api_keys(
    project_id="projectId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">create_api_key</a>(...) -> ApiKeyResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new API key for a project (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.create_api_key(
    project_id="projectId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**note:** `typing.Optional[str]` — Optional note for the API key
    
</dd>
</dl>

<dl>
<dd>

**public_key:** `typing.Optional[str]` — Optional predefined public key. Must start with 'pk-lf-'. If provided, secretKey must also be provided.
    
</dd>
</dl>

<dl>
<dd>

**secret_key:** `typing.Optional[str]` — Optional predefined secret key. Must start with 'sk-lf-'. If provided, publicKey must also be provided.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.projects.<a href="src/langfuse/projects/client.py">delete_api_key</a>(...) -> ApiKeyDeletionResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete an API key for a project (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.projects.delete_api_key(
    project_id="projectId",
    api_key_id="apiKeyId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**project_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**api_key_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## PromptVersion
<details><summary><code>client.prompt_version.<a href="src/langfuse/prompt_version/client.py">update</a>(...) -> Prompt</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update labels for a specific prompt version
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.prompt_version.update(
    name="name",
    version=1,
    new_labels=[
        "newLabels",
        "newLabels"
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` 

The name of the prompt. If the prompt is in a folder (e.g., "folder/subfolder/prompt-name"), 
the folder path must be URL encoded.
    
</dd>
</dl>

<dl>
<dd>

**version:** `int` — Version of the prompt to update
    
</dd>
</dl>

<dl>
<dd>

**new_labels:** `typing.List[str]` — New labels for the prompt version. Labels are unique across versions. The "latest" label is reserved and managed by Langfuse.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Prompts
<details><summary><code>client.prompts.<a href="src/langfuse/prompts/client.py">get</a>(...) -> Prompt</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a prompt
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.prompts.get(
    prompt_name="promptName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**prompt_name:** `str` 

The name of the prompt. If the prompt is in a folder (e.g., "folder/subfolder/prompt-name"), 
the folder path must be URL encoded.
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[int]` — Version of the prompt to be retrieved.
    
</dd>
</dl>

<dl>
<dd>

**label:** `typing.Optional[str]` — Label of the prompt to be retrieved. Defaults to "production" if no label or version is set.
    
</dd>
</dl>

<dl>
<dd>

**resolve:** `typing.Optional[bool]` — Resolve prompt dependencies before returning the prompt. Defaults to `true`. Set to `false` to return the raw stored prompt with dependency tags intact. This bypasses prompt caching and is intended for debugging or one-off jobs, not production runtime fetches.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/prompts/client.py">list</a>(...) -> PromptMetaListResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of prompt names with versions and labels
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.prompts.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**label:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**tag:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**from_updated_at:** `typing.Optional[datetime.datetime]` — Optional filter to only include prompt versions created/updated on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_updated_at:** `typing.Optional[datetime.datetime]` — Optional filter to only include prompt versions created/updated before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/prompts/client.py">create</a>(...) -> Prompt</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new version for the prompt with the given `name`
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI
from langfuse.prompts import CreateChatPromptRequest, ChatMessage

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.prompts.create(
    request=CreateChatPromptRequest(
        name="name",
        prompt=[
            ChatMessage(
                role="role",
                content="content",
            ),
            ChatMessage(
                role="role",
                content="content",
            )
        ],
        type="chat",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreatePromptRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/prompts/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete prompt versions. If neither version nor label is specified, all versions of the prompt are deleted.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.prompts.delete(
    prompt_name="promptName",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**prompt_name:** `str` — The name of the prompt
    
</dd>
</dl>

<dl>
<dd>

**label:** `typing.Optional[str]` — Optional label to filter deletion. If specified, deletes all prompt versions that have this label.
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[int]` — Optional version to filter deletion. If specified, deletes only this specific version of the prompt.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Scim
<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">get_service_provider_config</a>() -> ServiceProviderConfig</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get SCIM Service Provider Configuration (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.get_service_provider_config()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">get_resource_types</a>() -> ResourceTypesResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get SCIM Resource Types (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.get_resource_types()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">get_schemas</a>() -> SchemasResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get SCIM Schemas (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.get_schemas()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">list_users</a>(...) -> ScimUsersListResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List users in the organization (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.list_users()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**filter:** `typing.Optional[str]` — Filter expression (e.g. userName eq "value")
    
</dd>
</dl>

<dl>
<dd>

**start_index:** `typing.Optional[int]` — 1-based index of the first result to return (default 1)
    
</dd>
</dl>

<dl>
<dd>

**count:** `typing.Optional[int]` — Maximum number of results to return (default 100)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">create_user</a>(...) -> ScimUser</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new user in the organization (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI
from langfuse.scim import ScimName

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.create_user(
    user_name="userName",
    name=ScimName(),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**user_name:** `str` — User's email address (required)
    
</dd>
</dl>

<dl>
<dd>

**name:** `ScimName` — User's name information
    
</dd>
</dl>

<dl>
<dd>

**emails:** `typing.Optional[typing.List[ScimEmail]]` — User's email addresses
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` — Whether the user is active
    
</dd>
</dl>

<dl>
<dd>

**password:** `typing.Optional[str]` — Initial password for the user
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">get_user</a>(...) -> ScimUser</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a specific user by ID (requires organization-scoped API key)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.get_user(
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**user_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scim.<a href="src/langfuse/scim/client.py">delete_user</a>(...) -> EmptyResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove a user from the organization (requires organization-scoped API key). Note that this only removes the user from the organization but does not delete the user entity itself.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scim.delete_user(
    user_id="userId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**user_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## ScoreConfigs
<details><summary><code>client.score_configs.<a href="src/langfuse/score_configs/client.py">create</a>(...) -> ScoreConfig</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a score configuration (config). Score configs are used to define the structure of scores
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.score_configs.create(
    name="name",
    data_type="NUMERIC",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateScoreConfigRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_configs.<a href="src/langfuse/score_configs/client.py">get</a>(...) -> ScoreConfigs</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all score configs
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.score_configs.get()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_configs.<a href="src/langfuse/score_configs/client.py">get_by_id</a>(...) -> ScoreConfig</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a score config
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.score_configs.get_by_id(
    config_id="configId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**config_id:** `str` — The unique langfuse identifier of a score config
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_configs.<a href="src/langfuse/score_configs/client.py">update</a>(...) -> ScoreConfig</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a score config
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.score_configs.update(
    config_id="configId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**config_id:** `str` — The unique langfuse identifier of a score config
    
</dd>
</dl>

<dl>
<dd>

**request:** `UpdateScoreConfigRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Scores
<details><summary><code>client.scores.<a href="src/langfuse/scores/client.py">get_many</a>(...) -> GetScoresResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of scores (supports both trace and session scores)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scores.get_many()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` — Retrieve only scores with this userId associated to the trace.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Retrieve only scores with this name.
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include scores created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include scores created before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for scores where the environment is one of the provided values.
    
</dd>
</dl>

<dl>
<dd>

**source:** `typing.Optional[ScoreSource]` — Retrieve only scores from a specific source.
    
</dd>
</dl>

<dl>
<dd>

**operator:** `typing.Optional[str]` — Retrieve only scores with <operator> value.
    
</dd>
</dl>

<dl>
<dd>

**value:** `typing.Optional[float]` — Retrieve only scores with <operator> value.
    
</dd>
</dl>

<dl>
<dd>

**score_ids:** `typing.Optional[str]` — Comma-separated list of score IDs to limit the results to.
    
</dd>
</dl>

<dl>
<dd>

**config_id:** `typing.Optional[str]` — Retrieve only scores with a specific configId.
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` — Retrieve only scores with a specific sessionId.
    
</dd>
</dl>

<dl>
<dd>

**dataset_run_id:** `typing.Optional[str]` — Retrieve only scores with a specific datasetRunId.
    
</dd>
</dl>

<dl>
<dd>

**trace_id:** `typing.Optional[str]` — Retrieve only scores with a specific traceId.
    
</dd>
</dl>

<dl>
<dd>

**observation_id:** `typing.Optional[str]` — Comma-separated list of observation IDs to filter scores by.
    
</dd>
</dl>

<dl>
<dd>

**queue_id:** `typing.Optional[str]` — Retrieve only scores with a specific annotation queueId.
    
</dd>
</dl>

<dl>
<dd>

**data_type:** `typing.Optional[ScoreDataType]` — Retrieve only scores with a specific dataType.
    
</dd>
</dl>

<dl>
<dd>

**trace_tags:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Only scores linked to traces that include all of these tags will be returned.
    
</dd>
</dl>

<dl>
<dd>

**fields:** `typing.Optional[str]` — Comma-separated list of field groups to include in the response. Available field groups: 'score' (core score fields), 'trace' (trace properties: userId, tags, environment, sessionId). If not specified, both 'score' and 'trace' are returned by default. Example: 'score' to exclude trace data, 'score,trace' to include both. Note: When filtering by trace properties (using userId or traceTags parameters), the 'trace' field group must be included, otherwise a 400 error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**filter:** `typing.Optional[str]` — A JSON stringified array of filter objects. Each object requires type, column, operator, and value. Supports filtering by score metadata using the stringObject type. Example: [{"type":"stringObject","column":"metadata","key":"user_id","operator":"=","value":"abc123"}]. Supported types: stringObject (metadata key-value filtering), string, number, datetime, stringOptions, arrayOptions. Supported operators for stringObject: =, contains, does not contain, starts with, ends with.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.scores.<a href="src/langfuse/scores/client.py">get_by_id</a>(...) -> Score</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a score (supports both trace and session scores)
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.scores.get_by_id(
    score_id="scoreId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**score_id:** `str` — The unique langfuse identifier of a score
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Sessions
<details><summary><code>client.sessions.<a href="src/langfuse/sessions/client.py">list</a>(...) -> PaginatedSessions</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get sessions
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.sessions.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include sessions created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include sessions created before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for sessions where the environment is one of the provided values.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.sessions.<a href="src/langfuse/sessions/client.py">get</a>(...) -> SessionWithTraces</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a session. Please note that `traces` on this endpoint are not paginated, if you plan to fetch large sessions, consider `GET /api/public/traces?sessionId=<sessionId>`
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.sessions.get(
    session_id="sessionId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**session_id:** `str` — The unique id of a session
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Trace
<details><summary><code>client.trace.<a href="src/langfuse/trace/client.py">get</a>(...) -> TraceWithFullDetails</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a specific trace
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.trace.get(
    trace_id="traceId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**trace_id:** `str` — The unique langfuse identifier of a trace
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.trace.<a href="src/langfuse/trace/client.py">delete</a>(...) -> DeleteTraceResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a specific trace
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.trace.delete(
    trace_id="traceId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**trace_id:** `str` — The unique langfuse identifier of the trace to delete
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.trace.<a href="src/langfuse/trace/client.py">list</a>(...) -> Traces</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get list of traces
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.trace.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include traces with a trace.timestamp on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[datetime.datetime]` — Optional filter to only include traces with a trace.timestamp before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**order_by:** `typing.Optional[str]` — Format of the string [field].[asc/desc]. Fields: id, timestamp, name, userId, release, version, public, bookmarked, sessionId. Example: timestamp.asc
    
</dd>
</dl>

<dl>
<dd>

**tags:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Only traces that include all of these tags will be returned.
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[str]` — Optional filter to only include traces with a certain version.
    
</dd>
</dl>

<dl>
<dd>

**release:** `typing.Optional[str]` — Optional filter to only include traces with a certain release.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for traces where the environment is one of the provided values.
    
</dd>
</dl>

<dl>
<dd>

**fields:** `typing.Optional[str]` — Comma-separated list of fields to include in the response. Available field groups: 'core' (always included), 'io' (input, output, metadata), 'scores', 'observations', 'metrics'. If not specified, all fields are returned. Example: 'core,scores,metrics'. Note: Excluded 'observations' or 'scores' fields return empty arrays; excluded 'metrics' returns -1 for 'totalCost' and 'latency'.
    
</dd>
</dl>

<dl>
<dd>

**filter:** `typing.Optional[str]` 

JSON string containing an array of filter conditions. When provided, this takes precedence over query parameter filters (userId, name, sessionId, tags, version, release, environment, fromTimestamp, toTimestamp).

## Filter Structure
Each filter condition has the following structure:
```json
[
  {
    "type": string,           // Required. One of: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
    "column": string,         // Required. Column to filter on (see available columns below)
    "operator": string,       // Required. Operator based on type:
                              // - datetime: ">", "<", ">=", "<="
                              // - string: "=", "contains", "does not contain", "starts with", "ends with"
                              // - stringOptions: "any of", "none of"
                              // - categoryOptions: "any of", "none of"
                              // - arrayOptions: "any of", "none of", "all of"
                              // - number: "=", ">", "<", ">=", "<="
                              // - stringObject: "=", "contains", "does not contain", "starts with", "ends with"
                              // - numberObject: "=", ">", "<", ">=", "<="
                              // - boolean: "=", "<>"
                              // - null: "is null", "is not null"
    "value": any,             // Required (except for null type). Value to compare against. Type depends on filter type
    "key": string             // Required only for stringObject, numberObject, and categoryOptions types when filtering on nested fields like metadata
  }
]
```

## Available Columns

### Core Trace Fields
- `id` (string) - Trace ID
- `name` (string) - Trace name
- `timestamp` (datetime) - Trace timestamp
- `userId` (string) - User ID
- `sessionId` (string) - Session ID
- `environment` (string) - Environment tag
- `version` (string) - Version tag
- `release` (string) - Release tag
- `tags` (arrayOptions) - Array of tags
- `bookmarked` (boolean) - Bookmark status

### Structured Data
- `metadata` (stringObject/numberObject/categoryOptions) - Metadata key-value pairs. Use `key` parameter to filter on specific metadata keys.

### Aggregated Metrics (from observations)
These metrics are aggregated from all observations within the trace:
- `latency` (number) - Latency in seconds (time from first observation start to last observation end)
- `inputTokens` (number) - Total input tokens across all observations
- `outputTokens` (number) - Total output tokens across all observations
- `totalTokens` (number) - Total tokens (alias: `tokens`)
- `inputCost` (number) - Total input cost in USD
- `outputCost` (number) - Total output cost in USD
- `totalCost` (number) - Total cost in USD

### Observation Level Aggregations
These fields aggregate observation levels within the trace:
- `level` (string) - Highest severity level (ERROR > WARNING > DEFAULT > DEBUG)
- `warningCount` (number) - Count of WARNING level observations
- `errorCount` (number) - Count of ERROR level observations
- `defaultCount` (number) - Count of DEFAULT level observations
- `debugCount` (number) - Count of DEBUG level observations

### Scores (requires join with scores table)
- `scores_avg` (number) - Average of numeric scores (alias: `scores`)
- `score_categories` (categoryOptions) - Categorical score values

## Filter Examples
```json
[
  {
    "type": "datetime",
    "column": "timestamp",
    "operator": ">=",
    "value": "2024-01-01T00:00:00Z"
  },
  {
    "type": "string",
    "column": "userId",
    "operator": "=",
    "value": "user-123"
  },
  {
    "type": "number",
    "column": "totalCost",
    "operator": ">=",
    "value": 0.01
  },
  {
    "type": "arrayOptions",
    "column": "tags",
    "operator": "all of",
    "value": ["production", "critical"]
  },
  {
    "type": "stringObject",
    "column": "metadata",
    "key": "customer_tier",
    "operator": "=",
    "value": "enterprise"
  }
]
```

## Performance Notes
- Filtering on `userId`, `sessionId`, or `metadata` may enable skip indexes for better query performance
- Score filters require a join with the scores table and may impact query performance
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.trace.<a href="src/langfuse/trace/client.py">delete_multiple</a>(...) -> DeleteTraceResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete multiple traces
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import LangfuseAPI

client = LangfuseAPI(
    username="<username>",
    password="<password>",
    base_url="https://yourhost.com/path/to/api",
)

client.trace.delete_multiple(
    trace_ids=[
        "traceIds",
        "traceIds"
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**trace_ids:** `typing.List[str]` — List of trace IDs to delete
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

