import { z } from "zod";

export const ehrSchema = z.object({
  demographics: z.object({
    age: z.number().min(10).max(60),
    bmi: z.number().min(10).max(60),
  }),
  endocrine: z.object({
    amh: z.number().min(0).max(20),
    total_testosterone: z.number().min(0).max(300),
    hirsutism: z.number().min(0).max(36),
    acne_severity: z.number().min(0).max(4),
  }),
  menstrual: z.object({
    cycle_regularity: z.enum(["Regular", "Irregular"]),
    avg_cycle_length_days: z.number().min(15).max(60),
  }),
  metabolic: z.object({
    fasting_insulin: z.number().min(0).max(100),
    fasting_glucose: z.number().min(50).max(200),
  }),
});

export type EHRData = z.infer<typeof ehrSchema>;

export const defaultEhr: EHRData = {
  demographics: {
    age: 28,
    bmi: 28.5,
  },
  endocrine: {
    amh: 8.5,
    total_testosterone: 85,
    hirsutism: 12,
    acne_severity: 2,
  },
  menstrual: {
    cycle_regularity: "Irregular" as const,
    avg_cycle_length_days: 45,
  },
  metabolic: {
    fasting_insulin: 25,
    fasting_glucose: 105,
  },
};
