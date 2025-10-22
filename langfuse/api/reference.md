# Reference
## AnnotationQueues
<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">list_queues</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">create_queue</a>(...)</code></summary>
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
from langfuse import CreateAnnotationQueueRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.create_queue(
    request=CreateAnnotationQueueRequest(
        name="name",
        score_config_ids=["scoreConfigIds", "scoreConfigIds"],
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">get_queue</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">list_queue_items</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">get_queue_item</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">create_queue_item</a>(...)</code></summary>
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
from langfuse import AnnotationQueueObjectType, CreateAnnotationQueueItemRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.create_queue_item(
    queue_id="queueId",
    request=CreateAnnotationQueueItemRequest(
        object_id="objectId",
        object_type=AnnotationQueueObjectType.TRACE,
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">update_queue_item</a>(...)</code></summary>
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
from langfuse import UpdateAnnotationQueueItemRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.update_queue_item(
    queue_id="queueId",
    item_id="itemId",
    request=UpdateAnnotationQueueItemRequest(),
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">delete_queue_item</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">create_queue_assignment</a>(...)</code></summary>
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
from langfuse import AnnotationQueueAssignmentRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.create_queue_assignment(
    queue_id="queueId",
    request=AnnotationQueueAssignmentRequest(
        user_id="userId",
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

<details><summary><code>client.annotation_queues.<a href="src/langfuse/resources/annotation_queues/client.py">delete_queue_assignment</a>(...)</code></summary>
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
from langfuse import AnnotationQueueAssignmentRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.delete_queue_assignment(
    queue_id="queueId",
    request=AnnotationQueueAssignmentRequest(
        user_id="userId",
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
<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/resources/blob_storage_integrations/client.py">get_blob_storage_integrations</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/resources/blob_storage_integrations/client.py">upsert_blob_storage_integration</a>(...)</code></summary>
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
from langfuse import (
    BlobStorageExportFrequency,
    BlobStorageExportMode,
    BlobStorageIntegrationFileType,
    BlobStorageIntegrationType,
    CreateBlobStorageIntegrationRequest,
)
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.blob_storage_integrations.upsert_blob_storage_integration(
    request=CreateBlobStorageIntegrationRequest(
        project_id="projectId",
        type=BlobStorageIntegrationType.S_3,
        bucket_name="bucketName",
        region="region",
        export_frequency=BlobStorageExportFrequency.HOURLY,
        enabled=True,
        force_path_style=True,
        file_type=BlobStorageIntegrationFileType.JSON,
        export_mode=BlobStorageExportMode.FULL_HISTORY,
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

<details><summary><code>client.blob_storage_integrations.<a href="src/langfuse/resources/blob_storage_integrations/client.py">delete_blob_storage_integration</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateCommentRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.comments.create(
    request=CreateCommentRequest(
        project_id="projectId",
        object_type="objectType",
        object_id="objectId",
        content="content",
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

<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">get_by_id</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateDatasetItemRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_items.create(
    request=CreateDatasetItemRequest(
        dataset_name="datasetName",
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

<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get dataset items
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">delete</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.dataset_run_items.<a href="src/langfuse/resources/dataset_run_items/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateDatasetRunItemRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_run_items.create(
    request=CreateDatasetRunItemRequest(
        run_name="runName",
        dataset_item_id="datasetItemId",
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

<details><summary><code>client.dataset_run_items.<a href="src/langfuse/resources/dataset_run_items/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateDatasetRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.create(
    request=CreateDatasetRequest(
        name="name",
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

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get_run</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">delete_run</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get_runs</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.health.<a href="src/langfuse/resources/health/client.py">health</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.ingestion.<a href="src/langfuse/resources/ingestion/client.py">batch</a>(...)</code></summary>
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
from langfuse import IngestionEvent_ScoreCreate, ScoreBody
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.ingestion.batch(
    batch=[
        IngestionEvent_ScoreCreate(
            id="abcdef-1234-5678-90ab",
            timestamp="2022-01-01T00:00:00.000Z",
            body=ScoreBody(
                id="abcdef-1234-5678-90ab",
                trace_id="1234-5678-90ab-cdef",
                name="My Score",
                value=0.9,
                environment="default",
            ),
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

**batch:** `typing.Sequence[IngestionEvent]` — Batch of tracing events to be ingested. Discriminated by attribute `type`.
    
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

## LlmConnections
<details><summary><code>client.llm_connections.<a href="src/langfuse/resources/llm_connections/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.llm_connections.<a href="src/langfuse/resources/llm_connections/client.py">upsert</a>(...)</code></summary>
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
from langfuse import LlmAdapter, UpsertLlmConnectionRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.llm_connections.upsert(
    request=UpsertLlmConnectionRequest(
        provider="provider",
        adapter=LlmAdapter.ANTHROPIC,
        secret_key="secretKey",
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
<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">patch</a>(...)</code></summary>
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
import datetime

from langfuse import PatchMediaBody
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.media.patch(
    media_id="mediaId",
    request=PatchMediaBody(
        uploaded_at=datetime.datetime.fromisoformat(
            "2024-01-15 09:30:00+00:00",
        ),
        upload_http_status=1,
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

<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">get_upload_url</a>(...)</code></summary>
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
from langfuse import GetMediaUploadUrlRequest, MediaContentType
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.media.get_upload_url(
    request=GetMediaUploadUrlRequest(
        trace_id="traceId",
        content_type=MediaContentType.IMAGE_PNG,
        content_length=1,
        sha_256_hash="sha256Hash",
        field="field",
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
<details><summary><code>client.metrics.<a href="src/langfuse/resources/metrics/client.py">metrics</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get metrics from the Langfuse project using a query object.

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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

## Models
<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateModelRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.models.create(
    request=CreateModelRequest(
        model_name="modelName",
        match_pattern="matchPattern",
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

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">delete</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.observations.<a href="src/langfuse/resources/observations/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.observations.get(
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

<details><summary><code>client.observations.<a href="src/langfuse/resources/observations/client.py">get_many</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of observations
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**from_start_time:** `typing.Optional[dt.datetime]` — Retrieve only observations with a start_time on or after this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**to_start_time:** `typing.Optional[dt.datetime]` — Retrieve only observations with a start_time before this datetime (ISO 8601).
    
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

JSON string containing an array of filter conditions. When provided, this takes precedence over legacy filter parameters (userId, name, sessionId, tags, version, release, environment, fromTimestamp, toTimestamp).
Each filter condition has the following structure:
```json
[
  {
    "type": string,           // Required. One of: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
    "column": string,         // Required. Column to filter on
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
<details><summary><code>client.opentelemetry.<a href="src/langfuse/resources/opentelemetry/client.py">export_traces</a>(...)</code></summary>
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
from langfuse import (
    OtelAttribute,
    OtelAttributeValue,
    OtelResource,
    OtelResourceSpan,
    OtelScope,
    OtelScopeSpan,
    OtelSpan,
)
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
                    ),
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

**resource_spans:** `typing.Sequence[OtelResourceSpan]` — Array of resource spans containing trace data as defined in the OTLP specification
    
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
<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">get_organization_memberships</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">update_organization_membership</a>(...)</code></summary>
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
from langfuse import MembershipRequest, MembershipRole
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.organizations.update_organization_membership(
    request=MembershipRequest(
        user_id="userId",
        role=MembershipRole.OWNER,
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">delete_organization_membership</a>(...)</code></summary>
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
from langfuse import DeleteMembershipRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.organizations.delete_organization_membership(
    request=DeleteMembershipRequest(
        user_id="userId",
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">get_project_memberships</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">update_project_membership</a>(...)</code></summary>
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
from langfuse import MembershipRequest, MembershipRole
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.organizations.update_project_membership(
    project_id="projectId",
    request=MembershipRequest(
        user_id="userId",
        role=MembershipRole.OWNER,
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">delete_project_membership</a>(...)</code></summary>
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
from langfuse import DeleteMembershipRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.organizations.delete_project_membership(
    project_id="projectId",
    request=DeleteMembershipRequest(
        user_id="userId",
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

<details><summary><code>client.organizations.<a href="src/langfuse/resources/organizations/client.py">get_organization_projects</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

## Projects
<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">get</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get Project associated with API key
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">create</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">update</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.projects.update(
    project_id="projectId",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">delete</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">get_api_keys</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">create_api_key</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">delete_api_key</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.prompt_version.<a href="src/langfuse/resources/prompt_version/client.py">update</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompt_version.update(
    name="name",
    version=1,
    new_labels=["newLabels", "newLabels"],
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

**name:** `str` — The name of the prompt
    
</dd>
</dl>

<dl>
<dd>

**version:** `int` — Version of the prompt to update
    
</dd>
</dl>

<dl>
<dd>

**new_labels:** `typing.Sequence[str]` — New labels for the prompt version. Labels are unique across versions. The "latest" label is reserved and managed by Langfuse.
    
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
<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**prompt_name:** `str` — The name of the prompt
    
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

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**from_updated_at:** `typing.Optional[dt.datetime]` — Optional filter to only include prompt versions created/updated on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_updated_at:** `typing.Optional[dt.datetime]` — Optional filter to only include prompt versions created/updated before a certain datetime (ISO 8601)
    
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

<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">create</a>(...)</code></summary>
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
from langfuse import (
    ChatMessageWithPlaceholders_Chatmessage,
    CreatePromptRequest_Chat,
)
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompts.create(
    request=CreatePromptRequest_Chat(
        name="name",
        prompt=[
            ChatMessageWithPlaceholders_Chatmessage(
                role="role",
                content="content",
            ),
            ChatMessageWithPlaceholders_Chatmessage(
                role="role",
                content="content",
            ),
        ],
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

## Scim
<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">get_service_provider_config</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">get_resource_types</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">get_schemas</a>()</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">list_users</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">create_user</a>(...)</code></summary>
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
from langfuse import ScimName
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**emails:** `typing.Optional[typing.Sequence[ScimEmail]]` — User's email addresses
    
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">get_user</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.scim.<a href="src/langfuse/resources/scim/client.py">delete_user</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateScoreConfigRequest, ScoreDataType
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_configs.create(
    request=CreateScoreConfigRequest(
        name="name",
        data_type=ScoreDataType.NUMERIC,
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

<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">get_by_id</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">update</a>(...)</code></summary>
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
from langfuse import UpdateScoreConfigRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_configs.update(
    config_id="configId",
    request=UpdateScoreConfigRequest(),
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

## ScoreV2
<details><summary><code>client.score_v_2.<a href="src/langfuse/resources/score_v_2/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_v_2.get()

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

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include scores created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include scores created before a certain datetime (ISO 8601)
    
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

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_v_2.<a href="src/langfuse/resources/score_v_2/client.py">get_by_id</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_v_2.get_by_id(
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

## Score
<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">create</a>(...)</code></summary>
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
from langfuse import CreateScoreRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.create(
    request=CreateScoreRequest(
        name="name",
        value=1.1,
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

<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">delete</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.delete(
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
<details><summary><code>client.sessions.<a href="src/langfuse/resources/sessions/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include sessions created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include sessions created before a certain datetime (ISO 8601)
    
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

<details><summary><code>client.sessions.<a href="src/langfuse/resources/sessions/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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
<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">get</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">delete</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">list</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
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

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces with a trace.timestamp on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces with a trace.timestamp before a certain datetime (ISO 8601)
    
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

JSON string containing an array of filter conditions. When provided, this takes precedence over legacy filter parameters (userId, name, sessionId, tags, version, release, environment, fromTimestamp, toTimestamp).
Each filter condition has the following structure:
```json
[
  {
    "type": string,           // Required. One of: "datetime", "string", "number", "stringOptions", "categoryOptions", "arrayOptions", "stringObject", "numberObject", "boolean", "null"
    "column": string,         // Required. Column to filter on
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

<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">delete_multiple</a>(...)</code></summary>
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
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.trace.delete_multiple(
    trace_ids=["traceIds", "traceIds"],
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

**trace_ids:** `typing.Sequence[str]` — List of trace IDs to delete
    
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

