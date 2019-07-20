package com.salesforce.op.stages;

import java.lang.annotation.*;

/**
 * Stage of value class annotation to specify custom reader/writer implementation of [[ValueReaderWriter]].
 * Reader/writer implementation must extend [[ValueReaderWriter]] trait and has a single no arguments constructor.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@Inherited
public @interface ReaderWriter {

    /**
     * Reader/writer class extending [[ValueReaderWriter]] to use when reading/writing the stage or it's arguments.
     * It must extend [[ValueReaderWriter]] trait and has a single no arguments constructor.
     */
    Class<?> value();

}
