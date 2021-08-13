package io.github.singlerr.chat;

import discord4j.core.event.domain.Event;
import discord4j.core.event.domain.message.MessageCreateEvent;
import discord4j.core.object.reaction.ReactionEmoji;
import io.github.singlerr.api.event.EventListener;
import io.github.singlerr.api.event.EventManager;
import io.github.singlerr.api.module.Module;

public class DBMChatBot extends Module {
    @Override
    public void onEnable() {
        EventManager.getManager().registerEventListener(new EventListener<MessageCreateEvent>() {
            @Override
            public void on(MessageCreateEvent event) {

            }

            @Override
            public Class<MessageCreateEvent> getEventClass() {
                return MessageCreateEvent.class;
            }
        });
    }

    @Override
    public void onDisable() {

    }
}
