# อัปเดต repository เดิม

Repository เป้าหมาย:
https://github.com/ptakorn/ambiguity-guided-selective-oversampling

## แนะนำ: อัปเดตผ่าน GitHub Desktop หรือ Git ในเครื่อง

1. แตก ZIP จะได้โฟลเดอร์ ambiguity-guided-selective-oversampling
2. เปิด checkout/clone ของ repository เดิมในเครื่อง หากยังไม่มี ให้ clone repository เดิมก่อน
3. คัดลอก **เนื้อหาข้างใน** โฟลเดอร์จาก ZIP ไปทับไฟล์ใน checkout เดิม โดยรักษาโฟลเดอร์ย่อยทั้งหมดไว้ และอย่าลบโฟลเดอร์ .git ของ checkout
4. ตรวจรายการเปลี่ยนแปลง ไม่ต้องนำ vendor/WRND, .venv, downloaded dataset arrays หรือไฟล์ log ขึ้นไป
5. Commit เช่น `Add targeted and external SMOTE-ENN controls`
6. Push ไป repository เดิม
7. หากต้องการทำ release ให้ใช้ tag เช่น v2026.10.15 หลังตรวจไฟล์แล้ว
8. คัดลอก commit hash จริงหลังอัปโหลด ไปอัปเดต Code and Data Availability ในบทความ

ชุดนี้มีไฟล์จำนวนมาก จึงสะดวกกว่าการลากผ่านเว็บทีเดียว หากใช้เว็บและระบบจำกัดจำนวนไฟล์ ให้แบ่งอัปโหลดตามโฟลเดอร์ แต่รักษา README.md ไว้ที่ root ไม่ซ้อน folder ใหม่อีกชั้น

ถ้าใช้ Git CLI หลังคัดลอกไฟล์และตรวจ diff แล้ว:

```bash
git status
git add .
git diff --cached --stat
git commit -m "Add targeted and external SMOTE-ENN controls"
git push
```

ยังไม่มีการ push หรือสร้าง release จริงจากแพ็กเกจนี้ อย่าอ้าง commit เดิม 5c1850b ว่าครอบคลุมผลใหม่ และไม่ต้องใส่ DOI บทความที่ยังไม่ได้รับการกำหนด

แพ็กเกจนี้สร้างจากชุด repository ที่จัดไว้ก่อนหน้า ไม่ได้ดึงหรือ merge การแก้ไขล่าสุดบน remote ให้โดยอัตโนมัติ หากมีการแก้ไขบน GitHub ภายหลัง ให้ตรวจ diff ก่อนทับไฟล์
