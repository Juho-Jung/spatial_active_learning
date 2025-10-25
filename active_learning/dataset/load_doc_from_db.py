import mdb.document as mdb_d
import mdb.load as mdb_c
import utils.image_io as image_io


def _load_train_documents(target_lesion):
    """Load documents from train collection with consensus annotations."""
    collection = mdb_c.get_collection("sdc_ppm_train-0908", db_names=['cxr_new', 'projects', 'personal'])

    # Build query based on parameters
    query = {
        'is_pos': {'$in': [1]},
        'is_normal': {'$in': [0, 1]},
    }

    documents = list(collection.find(query).sort('_id', 1))  # Sort by _id for consistent ordering

    print(f"📊 Train Collection: {len(documents)}")
    return documents


def _create_split(documents, target_lesion, limit=None):
    """Create train/val split maintaining positive/negative ratios."""
    positive_docs = []
    negative_docs = []

    for doc in documents:
        objects = doc.get('objects', [])
        has_target_lesion = False
        for obj in objects:
            if obj.get('finding_name') == target_lesion:
                has_target_lesion = True
                break

        if has_target_lesion:
            positive_docs.append(doc)
        else:
            negative_docs.append(doc)

    documents = positive_docs

    # Apply limit if specified
    if limit is not None and len(documents) > limit:
        print(f"📊 Limiting dataset to {limit} samples (from {len(documents)})")
        documents = documents[:limit]

    return documents

if __name__ == "__main__":
    documents = _load_train_documents(target_lesion='pneumoperitoneum')
    documents = _create_split(documents, target_lesion='pneumoperitoneum', limit=None)
    print(len(documents))