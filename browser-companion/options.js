"use strict";

const input = document.getElementById("token");
const status = document.getElementById("status");
const save = document.getElementById("save");

void chrome.storage.local.get("token").then((saved) => {
  if (typeof saved.token === "string") input.value = saved.token;
});

save.addEventListener("click", async () => {
  const token = input.value.trim();
  if (token.length < 32 || !/^[A-Za-z0-9_-]+$/.test(token)) {
    status.textContent = "Token выглядит некорректно. Скопируй его из локальной команды.";
    return;
  }
  await chrome.storage.local.set({ token });
  void chrome.runtime.sendMessage({ type: "personal_agent_configuration_saved" }).catch(() => {});
  status.textContent = "Сохранено. Переключись на вкладку с обычной веб‑страницей.";
});
