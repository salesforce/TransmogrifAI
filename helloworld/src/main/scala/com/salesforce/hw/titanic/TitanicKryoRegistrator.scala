package com.salesforce.hw.titanic

import com.esotericsoftware.kryo.Kryo
import com.salesforce.op.utils.kryo.OpKryoRegistrator


class TitanicKryoRegistrator extends OpKryoRegistrator {

  override def registerCustomClasses(kryo: Kryo): Unit = {
    doClassRegistration(kryo)(classOf[TitanicFeatures])
    doAvroRegistration[com.salesforce.hw.titanic.Passenger](kryo)
  }

}
