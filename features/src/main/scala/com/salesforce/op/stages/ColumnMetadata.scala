package com.salesforce.op.stages

import org.apache.spark.sql.types.{Metadata, MetadataBuilder, MetadataWrapper, StructType}

/** A handler for Metadata objects used to store column metadata specifically. Using a Metadata object
 * for the column metadata allows the reuse of Metadata's JSON encoding to store the column metadata. */
object ColumnMetadata {

  /** An implicit class to insert column metadata into a spark schema (StructType) */
  implicit class SchemaWithColumnMetadata(schema: StructType) {
    /** inserts column metadata into a spark schema from a metadata object. If there's no metadata for given column,
     * nothing is inserted. */
    def insertColumnMetadata(columnMetadata: Metadata): StructType = {
      val metadataMap = new MetadataWrapper(columnMetadata).underlyingMap
      val fieldsWithMetadata = schema.map { case field =>
        metadataMap.get(field.name) match {
          case Some(metadata: Metadata) => field.copy(metadata = metadata)
          case _ => field
        }
      }
      StructType(fieldsWithMetadata)
    }

    /** Same as above but uses a similar signature as Map for convenience. */
    def insertColumnMetadata(elems: (String, Metadata)*): StructType = {
      insertColumnMetadata(ColumnMetadata.fromElems(elems: _*))
    }
  }

  /** Empty metadata object. */
  def empty: Metadata = Metadata.empty

  /** Extracts column metadata from a spark schema (StructType). */
  def fromSchema(schema: StructType): Metadata = {
    schema.fields.foldLeft(new MetadataBuilder()) { case (builder, field) =>
        builder.putMetadata(field.name, field.metadata)
      }.build()
  }

  /** Creates a new column metadata object using a similar signature as Map. */
  def fromElems(elems: (String, Metadata)*): Metadata = {
    elems.foldLeft(new MetadataBuilder()) { case (builder, (key, metadata)) =>
      builder.putMetadata(key, metadata)
    }.build()
  }
}
