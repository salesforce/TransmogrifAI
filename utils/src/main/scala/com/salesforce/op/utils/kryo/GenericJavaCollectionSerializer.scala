/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.kryo

import java.util.Collection

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.Input
import com.esotericsoftware.kryo.serializers.CollectionSerializer

/**
 * Special serializer for generic java collection types
 */
class GenericJavaCollectionSerializer[T <: Collection[_]](classType: Class[T]) extends CollectionSerializer {

  override def create(kryo: Kryo, input: Input, classType: Class[Collection[_]]): Collection[_] =
    kryo.newInstance(this.classType)

  override def createCopy(kryo: Kryo, original: Collection[_]): Collection[_] =
    kryo.newInstance(this.classType)

}
